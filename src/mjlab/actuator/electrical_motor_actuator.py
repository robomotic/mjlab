"""Electrical motor actuator with RL circuit and thermal dynamics.

This module provides an electrical motor actuator that implements realistic
electrical characteristics including:
- RL circuit dynamics (resistance, inductance, back-EMF)
- Voltage limiting (power supply constraints)
- Thermal modeling (I²R heating, temperature limits)
- Electrical state tracking (current, voltage, power, temperature)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg
from mjlab.actuator.inverter import InverterCfg
from mjlab.motor_database import MotorSpecification

if TYPE_CHECKING:
  from mjlab.entity import Entity

ElectricalMotorCfgT = TypeVar("ElectricalMotorCfgT", bound="ElectricalMotorActuatorCfg")


@dataclass(kw_only=True)
class ElectricalMotorActuatorCfg(DcMotorActuatorCfg):
  """Configuration for electrical motor actuator.

  Extends DcMotorActuator with realistic electrical and thermal characteristics:
  - RL circuit dynamics (voltage limiting, inductance lag)
  - Thermal modeling (I²R heating, temperature limits)
  - Electrical state tracking (current, voltage, power, temperature)

  The motor_spec provides all electrical/thermal parameters. The inherited
  saturation_effort and velocity_limit should match motor_spec values.
  """

  motor_spec: MotorSpecification
  """Motor specification with electrical/thermal/mechanical properties."""

  inverter_cfg: InverterCfg | None = None
  """Optional inverter configuration for AC motors (PMSM). When provided, models
  DC-to-AC conversion losses for PMSM motors (like Unitree Go2/H1).
  """

  def __post_init__(self) -> None:
    """Validate electrical motor parameters."""
    import warnings

    # Call parent validation (checks effort_limit vs saturation_effort)
    super().__post_init__()

    # Verify saturation_effort matches motor spec
    if abs(self.saturation_effort - self.motor_spec.peak_torque) > 1e-6:
      warnings.warn(
        f"saturation_effort ({self.saturation_effort}) does not match "
        f"motor_spec.peak_torque ({self.motor_spec.peak_torque}). "
        "Consider using motor_spec.peak_torque for consistency.",
        UserWarning,
        stacklevel=2,
      )

    # Verify velocity_limit matches motor spec
    if abs(self.velocity_limit - self.motor_spec.no_load_speed) > 1e-6:
      warnings.warn(
        f"velocity_limit ({self.velocity_limit}) does not match "
        f"motor_spec.no_load_speed ({self.motor_spec.no_load_speed}). "
        "Consider using motor_spec.no_load_speed for consistency.",
        UserWarning,
        stacklevel=2,
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> ElectricalMotorActuator:
    return ElectricalMotorActuator(self, entity, target_ids, target_names)


class ElectricalMotorActuator(
  DcMotorActuator[ElectricalMotorCfgT], Generic[ElectricalMotorCfgT]
):
  """Electrical motor actuator with RL circuit and thermal dynamics.

  This actuator extends DcMotorActuator with realistic electrical characteristics:
  - RL circuit dynamics: V = I·R + L·(dI/dt) + Ke·ω
  - Voltage limiting: V ∈ [V_min, V_max]
  - Thermal dynamics: dT/dt = (I²·R - (T-T_amb)/R_th) / τ_th
  - Temperature limits: T ∈ [T_amb, T_max]

  The electrical dynamics are computed per-step in compute(), while thermal
  dynamics are updated post-step in update().
  """

  def __init__(
    self,
    cfg: ElectricalMotorCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)

    # Electrical state tensors (initialized in initialize())
    self.current: torch.Tensor | None = None  # A
    self.voltage: torch.Tensor | None = None  # V
    self.back_emf: torch.Tensor | None = None  # V
    self.power_dissipation: torch.Tensor | None = None  # W

    # Thermal state
    self.winding_temperature: torch.Tensor | None = None  # °C

    # For RL integration (L·dI/dt term)
    self._previous_current: torch.Tensor | None = None  # A
    self._dt: float | None = None  # Simulation timestep

    # Motor constants from motor_spec (broadcasted tensors)
    self._motor_constant_kt: torch.Tensor | None = None  # N·m/A
    self._motor_constant_ke: torch.Tensor | None = None  # V·s/rad
    self._resistance: torch.Tensor | None = None  # Ω
    self._inductance: torch.Tensor | None = None  # H
    self._voltage_min: torch.Tensor | None = None  # V
    self._voltage_max: torch.Tensor | None = None  # V
    self._thermal_resistance: torch.Tensor | None = None  # °C/W
    self._thermal_time_constant: torch.Tensor | None = None  # s
    self._max_temperature: torch.Tensor | None = None  # °C
    self._ambient_temperature: torch.Tensor | None = None  # °C

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    # Store timestep
    self._dt = mj_model.opt.timestep

    num_envs = data.nworld
    num_joints = len(self._target_names)
    motor = self.cfg.motor_spec

    # Extract motor constants as tensors
    self._motor_constant_kt = torch.full(
      (num_envs, num_joints), motor.motor_constant_kt, device=device
    )
    self._motor_constant_ke = torch.full(
      (num_envs, num_joints), motor.motor_constant_ke, device=device
    )
    self._resistance = torch.full(
      (num_envs, num_joints), motor.resistance, device=device
    )
    self._inductance = torch.full(
      (num_envs, num_joints), motor.inductance, device=device
    )

    # Voltage limits
    self._voltage_min = torch.full(
      (num_envs, num_joints), motor.voltage_range[0], device=device
    )
    self._voltage_max = torch.full(
      (num_envs, num_joints), motor.voltage_range[1], device=device
    )

    # Thermal parameters
    self._thermal_resistance = torch.full(
      (num_envs, num_joints), motor.thermal_resistance, device=device
    )
    self._thermal_time_constant = torch.full(
      (num_envs, num_joints), motor.thermal_time_constant, device=device
    )
    self._max_temperature = torch.full(
      (num_envs, num_joints), motor.max_winding_temperature, device=device
    )
    self._ambient_temperature = torch.full(
      (num_envs, num_joints), motor.ambient_temperature, device=device
    )

    # Initialize electrical state
    self.current = torch.zeros(num_envs, num_joints, device=device)
    self.voltage = torch.zeros(num_envs, num_joints, device=device)
    self.back_emf = torch.zeros(num_envs, num_joints, device=device)
    self.power_dissipation = torch.zeros(num_envs, num_joints, device=device)
    self._previous_current = torch.zeros(num_envs, num_joints, device=device)

    # Initialize thermal state (start at ambient)
    self.winding_temperature = self._ambient_temperature.clone()

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    """Compute actuator torques with RL circuit dynamics."""
    assert self._motor_constant_kt is not None
    assert self._motor_constant_ke is not None
    assert self._resistance is not None
    assert self._inductance is not None
    assert self._voltage_min is not None
    assert self._voltage_max is not None
    assert self.current is not None
    assert self.voltage is not None
    assert self.back_emf is not None
    assert self._previous_current is not None
    assert self.stiffness is not None
    assert self.damping is not None
    assert self._dt is not None

    # Get timestep
    dt = self._dt

    # 1. Compute back-EMF from current joint velocity
    self.back_emf = self._motor_constant_ke * cmd.vel

    # 2. Compute desired torque from PD controller
    pos_error = cmd.position_target - cmd.pos
    vel_error = cmd.velocity_target - cmd.vel
    effort_desired = (
      self.stiffness * pos_error + self.damping * vel_error + cmd.effort_target
    )

    # 3. Required current for desired torque
    I_target = effort_desired / self._motor_constant_kt

    # 4. Terminal voltage for target current (semi-implicit)
    # V = I_target·R + L·(I_target - I_old)/dt + back_emf
    V_terminal = (
      I_target * self._resistance
      + self._inductance * (I_target - self._previous_current) / dt
      + self.back_emf
    )

    # 5. Voltage clamping (power supply limits)
    self.voltage = torch.clamp(V_terminal, min=self._voltage_min, max=self._voltage_max)

    # 6. Actual current from clamped voltage
    # V_clamped = I_actual·R + L·(I_actual - I_old)/dt + back_emf
    # Solve for I_actual:
    # I_actual = (V_clamped - back_emf + L·I_old/dt) / (R + L/dt)
    denominator = self._resistance + self._inductance / dt
    numerator = (
      self.voltage - self.back_emf + self._inductance * self._previous_current / dt
    )
    self.current = numerator / denominator

    # 7. Store previous current for next step
    self._previous_current = self.current.clone()

    # 8. Actual torque from actual current
    effort_actual = self._motor_constant_kt * self.current

    # 9. If inverter present, apply DC-to-AC conversion losses
    if self.cfg.inverter_cfg is not None:
      assert self.current is not None
      assert self.voltage is not None

      # Save AC-side current before modification
      i_ac = self.current

      # Compute load fraction: |I_motor| / I_stall
      load_fraction = torch.abs(i_ac) / self.cfg.motor_spec.stall_current

      # Get efficiency from inverter curve
      efficiency = self.cfg.inverter_cfg.get_efficiency(
        load_fraction, device=str(i_ac.device)
      )

      # DC-side current increased to account for inverter losses
      # P_dc = P_ac / η  =>  I_dc = I_ac / η
      # (assuming V_dc ≈ V_ac for power calculation)
      self.current = i_ac / efficiency

      # Additional power loss heats inverter (and motor)
      # ΔP = P_dc - P_ac = P_ac * (1/η - 1)
      # This adds to motor I²R losses
      p_inverter_loss = self.voltage * torch.abs(i_ac) * (1.0 - efficiency)

      # Add to power dissipation (affects thermal budget in update())
      # Note: This is computed here but dissipation is updated in update()
      # Store for later use
      if not hasattr(self, "_inverter_loss"):
        self._inverter_loss = torch.zeros_like(i_ac)
      self._inverter_loss = p_inverter_loss

    # 10. Apply DC motor torque-speed curve (parent class)
    # This calls _clip_effort() which applies torque-speed limiting
    return self._clip_effort(effort_actual)

  def update(self, dt: float) -> None:
    """Update thermal dynamics post-step."""
    # Call parent update first
    super().update(dt)

    assert self.current is not None
    assert self._resistance is not None
    assert self.winding_temperature is not None
    assert self._ambient_temperature is not None
    assert self._thermal_resistance is not None
    assert self._thermal_time_constant is not None
    assert self._max_temperature is not None
    assert self.power_dissipation is not None

    # Compute power dissipation (I²R)
    self.power_dissipation = self.current**2 * self._resistance

    # Add inverter losses if present
    if self.cfg.inverter_cfg is not None and hasattr(self, "_inverter_loss"):
      self.power_dissipation = self.power_dissipation + self._inverter_loss

    # Thermal dynamics: dT/dt = (P_loss - (T-T_amb)/R_th) / τ_th
    heat_in = self.power_dissipation
    heat_out = (
      self.winding_temperature - self._ambient_temperature
    ) / self._thermal_resistance

    dT_dt = (heat_in - heat_out) / self._thermal_time_constant
    self.winding_temperature = self.winding_temperature + dT_dt * dt

    # Temperature clamping (safety limit)
    self.winding_temperature = torch.clamp(
      self.winding_temperature,
      min=self._ambient_temperature,
      max=self._max_temperature,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset electrical state for specified environments."""
    assert self.current is not None
    assert self._previous_current is not None
    assert self.voltage is not None
    assert self.back_emf is not None
    assert self.power_dissipation is not None
    assert self.winding_temperature is not None
    assert self._ambient_temperature is not None

    if env_ids is None:
      # Reset all environments
      self.current.zero_()
      self._previous_current.zero_()
      self.voltage.zero_()
      self.back_emf.zero_()
      self.power_dissipation.zero_()
      self.winding_temperature[:] = self._ambient_temperature
    else:
      # Reset specific environments
      self.current[env_ids] = 0.0
      self._previous_current[env_ids] = 0.0
      self.voltage[env_ids] = 0.0
      self.back_emf[env_ids] = 0.0
      self.power_dissipation[env_ids] = 0.0
      self.winding_temperature[env_ids] = self._ambient_temperature[env_ids]

    # Call parent reset (resets PD state)
    super().reset(env_ids)
