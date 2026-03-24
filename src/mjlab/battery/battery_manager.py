"""Battery manager for scene-level power management.

This module provides scene-level battery state tracking and power distribution
management. The BatteryManager aggregates power draw from all electrical
actuators and simulates realistic battery behavior including voltage sag,
state-of-charge tracking, and thermal dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.battery_database import BatterySpecification

if TYPE_CHECKING:
  from mjlab.scene import Scene


@dataclass
class BatteryManagerCfg:
  """Configuration for scene-level battery management.

  The battery manager tracks battery state across parallel environments and
  provides dynamic voltage feedback to electrical motor actuators.
  """

  battery_spec: BatterySpecification
  """Battery specification from battery database."""

  entity_names: tuple[str, ...] = ("robot",)
  """Names of entities powered by this battery."""

  initial_soc: float = 1.0
  """Initial state of charge (0.0 to 1.0). Default is fully charged."""

  enable_voltage_feedback: bool = True
  """Whether to dynamically update motor voltage limits based on battery state."""

  allow_regenerative_braking: bool = False
  """Whether to allow motor current to flow back into battery during braking.

  When motors are backdriven (e.g., by gravity), they act as generators
  and produce negative current. This flag controls whether that current
  is accepted by the battery:

  - False (default): Battery rejects backfeed; negative current clamped to zero.
    Energy dissipates as heat in motor windings. Realistic for most commercial
    batteries (Li-Po, Li-ion) that lack charging circuits.

  - True: Motors can return energy to battery (regenerative braking). Battery
    SOC increases when backdriven. Only use for battery specs that explicitly
    support regenerative charging (e.g., future LiFePO4 models with charge
    controller).

  Default is False for backwards compatibility and realistic simulation.
  """


class BatteryManager:
  """Manages battery state and power distribution for a scene.

  Tracks battery state-of-charge (SOC), terminal voltage, temperature, and
  current draw across parallel environments. Aggregates power consumption from
  all electrical actuators and feeds terminal voltage back to actuators for
  realistic voltage sag under load.

  State tensors are all shape (num_envs,) for per-environment tracking.

  Physics:
      V_terminal = V_oc(SOC) - I * R_internal(SOC, T)
      dSOC/dt = -I / (Q_capacity * 3600)
      dT/dt = (I²R - (T-T_amb)/R_th) / τ_th
  """

  def __init__(self, cfg: BatteryManagerCfg, scene: Scene) -> None:
    """Initialize battery manager.

    Args:
        cfg: Battery manager configuration
        scene: Scene instance for accessing entities and actuators
    """
    self.cfg = cfg
    self.scene = scene

    # Battery state tensors (initialized in initialize())
    self.soc: torch.Tensor | None = None
    """State of charge [0, 1] for each environment."""

    self.voltage: torch.Tensor | None = None
    """Terminal voltage (V) for each environment."""

    self.current: torch.Tensor | None = None
    """Total current draw (A) for each environment."""

    self.temperature: torch.Tensor | None = None
    """Battery temperature (°C) for each environment."""

    self.power_out: torch.Tensor | None = None
    """Output power (W) for each environment."""

    self.internal_resistance: torch.Tensor | None = None
    """Internal resistance (Ω) for each environment (varies with SOC and T)."""

    # Device tracking
    self._device: str | None = None

  def initialize(self, num_envs: int, device: str) -> None:
    """Initialize battery state tensors.

    Args:
        num_envs: Number of parallel environments
        device: Torch device (e.g., "cuda", "cpu")
    """
    self._device = device

    # Initialize SOC
    self.soc = torch.full(
      (num_envs,), self.cfg.initial_soc, device=device, dtype=torch.float32
    )

    # Initialize voltage (start at nominal voltage)
    self.voltage = torch.full(
      (num_envs,),
      self.cfg.battery_spec.nominal_voltage,
      device=device,
      dtype=torch.float32,
    )

    # Initialize current (zero at start)
    self.current = torch.zeros(num_envs, device=device, dtype=torch.float32)

    # Initialize temperature (ambient)
    self.temperature = torch.full(
      (num_envs,),
      self.cfg.battery_spec.ambient_temperature,
      device=device,
      dtype=torch.float32,
    )

    # Initialize power output
    self.power_out = torch.zeros(num_envs, device=device, dtype=torch.float32)

    # Initialize internal resistance (base value)
    self.internal_resistance = torch.full(
      (num_envs,),
      self.cfg.battery_spec.internal_resistance,
      device=device,
      dtype=torch.float32,
    )

  def compute_voltage(self) -> torch.Tensor:
    """Compute terminal voltage based on current SOC and load.

    Called BEFORE actuator compute() to provide voltage limits. Uses open-
    circuit voltage curve and internal resistance to model voltage sag.

    Returns:
        Terminal voltage (V) for each environment, shape (num_envs,)
    """
    assert self.soc is not None
    assert self.current is not None
    assert self.voltage is not None
    assert self.internal_resistance is not None
    assert self._device is not None

    # 1. Get open-circuit voltage from SOC
    v_oc = self._interpolate_ocv(self.soc)

    # 2. Update internal resistance (SOC + temperature dependence)
    self._update_internal_resistance()

    # 3. Voltage drop from load: V_drop = I * R
    v_drop = self.current * self.internal_resistance

    # 4. Terminal voltage (clamped to battery limits)
    self.voltage = torch.clamp(
      v_oc - v_drop,
      min=self.cfg.battery_spec.min_voltage,
      max=self.cfg.battery_spec.max_voltage,
    )

    return self.voltage

  def aggregate_current(self) -> torch.Tensor:
    """Aggregate total current draw from all actuators.

    Called AFTER actuator compute() to sum up power draw from all electrical
    motors. Only considers ElectricalMotorActuator instances.

    Returns:
        Total current (A) for each environment, shape (num_envs,)
    """
    assert self.current is not None

    # Import here to avoid circular dependency
    from mjlab.actuator import ElectricalMotorActuator

    # Reset current
    self.current.zero_()

    # Aggregate from all entities
    for entity_name in self.cfg.entity_names:
      if entity_name not in self.scene.entities:
        continue

      entity = self.scene.entities[entity_name]

      # Sum current from all electrical motor actuators
      for actuator in entity._actuators:
        if isinstance(actuator, ElectricalMotorActuator):
          assert actuator.current is not None
          # Sum across all joints (dimension 1)
          self.current += actuator.current.sum(dim=1)

    # Clamp negative current if regenerative braking disabled
    if not self.cfg.allow_regenerative_braking:
      # Reject negative current (no energy return to battery)
      self.current = torch.clamp(self.current, min=0.0)

    # Clamp to battery maximum continuous current
    self.current = torch.clamp(
      self.current, max=self.cfg.battery_spec.max_continuous_current
    )

    return self.current

  def update(self, dt: float) -> None:
    """Update battery state (SOC, temperature) after physics step.

    Args:
        dt: Time step duration (seconds)
    """
    assert self.soc is not None
    assert self.current is not None
    assert self.temperature is not None
    assert self.internal_resistance is not None
    assert self.power_out is not None
    assert self.voltage is not None

    # 1. Update SOC: dSOC/dt = -I / (Q * 3600)
    capacity_as = self.cfg.battery_spec.capacity_ah * 3600.0
    dsoc_dt = -self.current / capacity_as

    self.soc = torch.clamp(
      self.soc + dsoc_dt * dt,
      min=self.cfg.battery_spec.min_soc,
      max=self.cfg.battery_spec.max_soc,
    )

    # 2. Thermal dynamics
    # Power loss: P = I²R
    power_loss = self.current**2 * self.internal_resistance

    # Heat flows: in (I²R) and out (convection)
    heat_in = power_loss
    heat_out = (
      self.temperature - self.cfg.battery_spec.ambient_temperature
    ) / self.cfg.battery_spec.thermal_resistance

    # Thermal time constant: τ = C_th * R_th
    tau_th = (
      self.cfg.battery_spec.thermal_capacity * self.cfg.battery_spec.thermal_resistance
    )

    # Temperature update: dT/dt = (P_in - P_out) / τ_th
    dT_dt = (heat_in - heat_out) / tau_th

    self.temperature = torch.clamp(
      self.temperature + dT_dt * dt,
      min=self.cfg.battery_spec.min_temperature,
      max=self.cfg.battery_spec.max_temperature,
    )

    # 3. Update power output
    self.power_out = self.voltage * self.current

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset battery state for specified environments.

    Args:
        env_ids: Environment indices to reset, or None for all
    """
    assert self.soc is not None
    assert self.voltage is not None
    assert self.current is not None
    assert self.temperature is not None
    assert self.power_out is not None

    if env_ids is None:
      # Reset all environments
      self.soc[:] = self.cfg.initial_soc
      self.voltage[:] = self.cfg.battery_spec.nominal_voltage
      self.current.zero_()
      self.temperature[:] = self.cfg.battery_spec.ambient_temperature
      self.power_out.zero_()
    else:
      # Reset specific environments
      self.soc[env_ids] = self.cfg.initial_soc
      self.voltage[env_ids] = self.cfg.battery_spec.nominal_voltage
      self.current[env_ids] = 0.0
      self.temperature[env_ids] = self.cfg.battery_spec.ambient_temperature
      self.power_out[env_ids] = 0.0

  def _interpolate_ocv(self, soc: torch.Tensor) -> torch.Tensor:
    """Interpolate open-circuit voltage from SOC curve.

    Args:
        soc: State of charge [0, 1], shape (num_envs,)

    Returns:
        Pack voltage (V), shape (num_envs,)
    """
    assert self.cfg.battery_spec.ocv_curve is not None
    assert self._device is not None

    # Extract SOC and voltage arrays from curve
    soc_points = torch.tensor(
      [p[0] for p in self.cfg.battery_spec.ocv_curve],
      device=self._device,
      dtype=torch.float32,
    )
    voltage_points = torch.tensor(
      [p[1] for p in self.cfg.battery_spec.ocv_curve],
      device=self._device,
      dtype=torch.float32,
    )

    # Linear interpolation
    # Clamp SOC to curve range
    soc_clamped = torch.clamp(soc, min=soc_points[0], max=soc_points[-1])

    # Find interpolation indices
    indices = torch.searchsorted(soc_points, soc_clamped, right=False)
    indices = torch.clamp(indices, min=1, max=len(soc_points) - 1)

    # Interpolate
    soc_lo = soc_points[indices - 1]
    soc_hi = soc_points[indices]
    v_lo = voltage_points[indices - 1]
    v_hi = voltage_points[indices]

    # Linear interpolation: v = v_lo + (soc - soc_lo) * (v_hi - v_lo) / (soc_hi - soc_lo)
    alpha = (soc_clamped - soc_lo) / (soc_hi - soc_lo + 1e-8)
    v_cell = v_lo + alpha * (v_hi - v_lo)

    # Convert to pack voltage
    v_pack = v_cell * self.cfg.battery_spec.cells_series

    return v_pack

  def _update_internal_resistance(self) -> None:
    """Update internal resistance based on SOC and temperature.

    Updates self.internal_resistance in-place using:
        R = R_base * f(SOC) * (1 + α * ΔT)

    where f(SOC) is from resistance-SOC curve and α is temp coefficient.
    """
    assert self.soc is not None
    assert self.temperature is not None
    assert self.internal_resistance is not None
    assert self.cfg.battery_spec.internal_resistance_soc_curve is not None
    assert self._device is not None

    # 1. Get SOC-dependent multiplier
    soc_points = torch.tensor(
      [p[0] for p in self.cfg.battery_spec.internal_resistance_soc_curve],
      device=self._device,
      dtype=torch.float32,
    )
    r_mult_points = torch.tensor(
      [p[1] for p in self.cfg.battery_spec.internal_resistance_soc_curve],
      device=self._device,
      dtype=torch.float32,
    )

    # Interpolate resistance multiplier
    soc_clamped = torch.clamp(self.soc, min=soc_points[0], max=soc_points[-1])
    indices = torch.searchsorted(soc_points, soc_clamped, right=False)
    indices = torch.clamp(indices, min=1, max=len(soc_points) - 1)

    soc_lo = soc_points[indices - 1]
    soc_hi = soc_points[indices]
    r_mult_lo = r_mult_points[indices - 1]
    r_mult_hi = r_mult_points[indices]

    alpha = (soc_clamped - soc_lo) / (soc_hi - soc_lo + 1e-8)
    r_multiplier = r_mult_lo + alpha * (r_mult_hi - r_mult_lo)

    # 2. Temperature dependence: (1 + α * ΔT)
    temp_delta = self.temperature - self.cfg.battery_spec.ambient_temperature
    temp_factor = (
      1.0 + self.cfg.battery_spec.internal_resistance_temp_coeff * temp_delta
    )

    # 3. Final resistance
    self.internal_resistance[:] = (
      self.cfg.battery_spec.internal_resistance * r_multiplier * temp_factor
    )
