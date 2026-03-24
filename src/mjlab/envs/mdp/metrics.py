from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import MetricsTermCfg


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean absolute action acceleration.

  Lower values indicate smoother actions.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  # Discrete second derivative: a_t - 2 * a_{t-1} + a_{t-2}.  (B, N)
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.mean(torch.abs(action_acc), dim=-1)  # (B,)


# Motor Metrics


def motor_current_avg(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Average motor current across all electrical motors.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.

  Returns:
    Per-environment average current in amperes. Shape: ``(B,)``.
    Returns zeros if no electrical motors are present.
  """
  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  # Find all electrical motor actuators
  currents = []
  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      # actuator.current is (num_envs, num_joints)
      currents.append(actuator.current)

  if not currents:
    return torch.zeros(env.num_envs, device=env.device)

  # Stack and average across all motors
  all_currents = torch.cat(currents, dim=1)  # (B, total_motors)
  return torch.mean(torch.abs(all_currents), dim=1)  # (B,)


def motor_voltage_avg(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Average motor voltage across all electrical motors.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.

  Returns:
    Per-environment average voltage in volts. Shape: ``(B,)``.
    Returns zeros if no electrical motors are present.
  """
  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  voltages = []
  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      voltages.append(actuator.voltage)

  if not voltages:
    return torch.zeros(env.num_envs, device=env.device)

  all_voltages = torch.cat(voltages, dim=1)
  return torch.mean(torch.abs(all_voltages), dim=1)


def motor_power_total(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Total motor power dissipation (I²R losses) across all electrical motors.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.

  Returns:
    Per-environment total power dissipation in watts. Shape: ``(B,)``.
    Returns zeros if no electrical motors are present.
  """
  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  power_values = []
  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      power_values.append(actuator.power_dissipation)

  if not power_values:
    return torch.zeros(env.num_envs, device=env.device)

  all_power = torch.cat(power_values, dim=1)
  return torch.sum(all_power, dim=1)


def motor_temperature_max(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Maximum winding temperature across all electrical motors.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.

  Returns:
    Per-environment maximum temperature in degrees Celsius. Shape: ``(B,)``.
    Returns zeros if no electrical motors are present.
  """
  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  temperatures = []
  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      temperatures.append(actuator.winding_temperature)

  if not temperatures:
    return torch.zeros(env.num_envs, device=env.device)

  all_temps = torch.cat(temperatures, dim=1)
  return torch.max(all_temps, dim=1)[0]


def motor_back_emf_avg(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Average back-EMF voltage across all electrical motors.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.

  Returns:
    Per-environment average back-EMF in volts. Shape: ``(B,)``.
    Returns zeros if no electrical motors are present.
  """
  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  back_emfs = []
  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      back_emfs.append(actuator.back_emf)

  if not back_emfs:
    return torch.zeros(env.num_envs, device=env.device)

  all_back_emf = torch.cat(back_emfs, dim=1)
  return torch.mean(torch.abs(all_back_emf), dim=1)


# Per-Joint Motor Metrics


def motor_current_joint(
  env: ManagerBasedRlEnv, entity_name: str = "robot", joint_name: str | None = None
) -> torch.Tensor:
  """Motor current for a specific joint.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.
    joint_name: Name of the joint to monitor. If None, returns zeros.

  Returns:
    Per-environment current in amperes for the specified joint. Shape: ``(B,)``.
    Returns zeros if joint not found or not an electrical motor.
  """
  if joint_name is None:
    return torch.zeros(env.num_envs, device=env.device)

  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  # Find the actuator controlling this joint
  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      # Check if this joint is in the actuator's target names
      if hasattr(actuator, "_target_names") and joint_name in actuator._target_names:
        joint_idx = actuator._target_names.index(joint_name)
        assert actuator.current is not None
        return torch.abs(actuator.current[:, joint_idx])

  # Joint not found or not electrical
  return torch.zeros(env.num_envs, device=env.device)


def motor_voltage_joint(
  env: ManagerBasedRlEnv, entity_name: str = "robot", joint_name: str | None = None
) -> torch.Tensor:
  """Motor voltage for a specific joint.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.
    joint_name: Name of the joint to monitor. If None, returns zeros.

  Returns:
    Per-environment voltage in volts for the specified joint. Shape: ``(B,)``.
    Returns zeros if joint not found or not an electrical motor.
  """
  if joint_name is None:
    return torch.zeros(env.num_envs, device=env.device)

  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      if hasattr(actuator, "_target_names") and joint_name in actuator._target_names:
        joint_idx = actuator._target_names.index(joint_name)
        assert actuator.voltage is not None
        return torch.abs(actuator.voltage[:, joint_idx])

  return torch.zeros(env.num_envs, device=env.device)


def motor_power_joint(
  env: ManagerBasedRlEnv, entity_name: str = "robot", joint_name: str | None = None
) -> torch.Tensor:
  """Motor power dissipation for a specific joint.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.
    joint_name: Name of the joint to monitor. If None, returns zeros.

  Returns:
    Per-environment power dissipation in watts for the specified joint. Shape: ``(B,)``.
    Returns zeros if joint not found or not an electrical motor.
  """
  if joint_name is None:
    return torch.zeros(env.num_envs, device=env.device)

  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      if hasattr(actuator, "_target_names") and joint_name in actuator._target_names:
        joint_idx = actuator._target_names.index(joint_name)
        assert actuator.power_dissipation is not None
        return actuator.power_dissipation[:, joint_idx]

  return torch.zeros(env.num_envs, device=env.device)


def motor_temperature_joint(
  env: ManagerBasedRlEnv, entity_name: str = "robot", joint_name: str | None = None
) -> torch.Tensor:
  """Motor winding temperature for a specific joint.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.
    joint_name: Name of the joint to monitor. If None, returns zeros.

  Returns:
    Per-environment temperature in degrees Celsius for the specified joint. Shape: ``(B,)``.
    Returns zeros if joint not found or not an electrical motor.
  """
  if joint_name is None:
    return torch.zeros(env.num_envs, device=env.device)

  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      if hasattr(actuator, "_target_names") and joint_name in actuator._target_names:
        joint_idx = actuator._target_names.index(joint_name)
        assert actuator.winding_temperature is not None
        return actuator.winding_temperature[:, joint_idx]

  return torch.zeros(env.num_envs, device=env.device)


def motor_back_emf_joint(
  env: ManagerBasedRlEnv, entity_name: str = "robot", joint_name: str | None = None
) -> torch.Tensor:
  """Motor back-EMF for a specific joint.

  Args:
    env: The RL environment.
    entity_name: Name of the entity containing electrical motors.
    joint_name: Name of the joint to monitor. If None, returns zeros.

  Returns:
    Per-environment back-EMF in volts for the specified joint. Shape: ``(B,)``.
    Returns zeros if joint not found or not an electrical motor.
  """
  if joint_name is None:
    return torch.zeros(env.num_envs, device=env.device)

  entity = env.scene.entities.get(entity_name)
  if entity is None:
    return torch.zeros(env.num_envs, device=env.device)

  from mjlab.actuator import ElectricalMotorActuator

  for actuator in entity._actuators:
    if isinstance(actuator, ElectricalMotorActuator):
      if hasattr(actuator, "_target_names") and joint_name in actuator._target_names:
        joint_idx = actuator._target_names.index(joint_name)
        assert actuator.back_emf is not None
        return torch.abs(actuator.back_emf[:, joint_idx])

  return torch.zeros(env.num_envs, device=env.device)


# Battery Metrics


def battery_soc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Battery state of charge (0-1 scale).

  Args:
    env: The RL environment.

  Returns:
    Per-environment SOC in range [0, 1]. Shape: ``(B,)``.
    Returns zeros if no battery is present.
  """
  battery = env.scene._battery_manager
  if battery is None:
    return torch.zeros(env.num_envs, device=env.device)
  assert battery.soc is not None
  return battery.soc


def battery_voltage(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Battery terminal voltage.

  Args:
    env: The RL environment.

  Returns:
    Per-environment voltage in volts. Shape: ``(B,)``.
    Returns zeros if no battery is present.
  """
  battery = env.scene._battery_manager
  if battery is None:
    return torch.zeros(env.num_envs, device=env.device)
  assert battery.voltage is not None
  return battery.voltage


def battery_current(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Battery output current (total draw from all motors).

  Args:
    env: The RL environment.

  Returns:
    Per-environment current in amperes. Shape: ``(B,)``.
    Returns zeros if no battery is present.
  """
  battery = env.scene._battery_manager
  if battery is None:
    return torch.zeros(env.num_envs, device=env.device)
  assert battery.current is not None
  return battery.current


def battery_power(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Battery output power.

  Args:
    env: The RL environment.

  Returns:
    Per-environment power in watts. Shape: ``(B,)``.
    Returns zeros if no battery is present.
  """
  battery = env.scene._battery_manager
  if battery is None:
    return torch.zeros(env.num_envs, device=env.device)
  assert battery.power_out is not None
  return battery.power_out


def battery_temperature(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Battery temperature.

  Args:
    env: The RL environment.

  Returns:
    Per-environment temperature in degrees Celsius. Shape: ``(B,)``.
    Returns zeros if no battery is present.
  """
  battery = env.scene._battery_manager
  if battery is None:
    return torch.zeros(env.num_envs, device=env.device)
  assert battery.temperature is not None
  return battery.temperature


# Class-Based Cumulative Metrics


class CumulativeEnergyMetric:
  """Cumulative electrical energy consumption metric.

  Tracks total electrical energy consumed from battery over an episode.
  Resets to zero at episode start and accumulates energy each step.

  The metric computes: E_total = ∫ P_battery * dt (in watt-hours)

  Example:
    Add to environment config::

      from mjlab.envs.mdp.metrics import CumulativeEnergyMetric
      from mjlab.managers import MetricsTermCfg

      cfg.metrics = {
          "energy_consumed_wh": MetricsTermCfg(func=CumulativeEnergyMetric()),
      }
  """

  def __init__(self):
    """Initialize cumulative energy metric."""
    self._cumulative_energy: torch.Tensor | None = None
    self._num_envs: int = 0
    self._device: str = "cpu"
    self._dt: float = 0.0

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Compute cumulative energy consumption.

    Args:
      env: The RL environment.

    Returns:
      Per-environment cumulative energy in watt-hours. Shape: ``(B,)``.
      Returns zeros if no battery is present.
    """
    # Initialize on first call
    if self._cumulative_energy is None:
      self._num_envs = env.num_envs
      self._device = env.device
      self._dt = env.physics_dt * env.cfg.decimation  # Environment step time
      self._cumulative_energy = torch.zeros(
        self._num_envs, device=self._device, dtype=torch.float32
      )

    # Get battery power
    battery = env.scene._battery_manager
    if battery is None or battery.power_out is None:
      return self._cumulative_energy

    # Accumulate energy: E += P * dt (convert to Wh: divide by 3600)
    power_watts = battery.power_out  # (B,)
    energy_increment_wh = power_watts * self._dt / 3600.0  # Convert J to Wh
    self._cumulative_energy += energy_increment_wh

    return self._cumulative_energy

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    """Reset cumulative energy for specified environments.

    Args:
      env_ids: Environment indices to reset. If None, resets all.

    Returns:
      Dictionary with final energy values before reset (for logging).
    """
    if self._cumulative_energy is None:
      return {}

    if env_ids is None:
      # Reset all environments
      final_values = {
        f"final_energy_wh_env_{i}": self._cumulative_energy[i].item()
        for i in range(self._num_envs)
      }
      self._cumulative_energy.zero_()
    else:
      # Reset specific environments
      final_values = {
        f"final_energy_wh_env_{i}": self._cumulative_energy[i].item()
        for i in env_ids.tolist()
      }
      self._cumulative_energy[env_ids] = 0.0

    return final_values


class CumulativeMechanicalWorkMetric:
  """Cumulative mechanical work output metric.

  Tracks total mechanical work performed by all motors over an episode.
  Resets to zero at episode start and accumulates work each step.

  The metric computes: W_total = ∫ τ * ω * dt (in joules)

  Example:
    Add to environment config::

      from mjlab.envs.mdp.metrics import CumulativeMechanicalWorkMetric
      from mjlab.managers import MetricsTermCfg

      cfg.metrics = {
          "mechanical_work_j": MetricsTermCfg(func=CumulativeMechanicalWorkMetric()),
      }
  """

  def __init__(self, entity_name: str = "robot"):
    """Initialize cumulative mechanical work metric.

    Args:
      entity_name: Name of the entity containing electrical motors.
    """
    self._entity_name = entity_name
    self._cumulative_work: torch.Tensor | None = None
    self._num_envs: int = 0
    self._device: str = "cpu"
    self._dt: float = 0.0

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Compute cumulative mechanical work.

    Args:
      env: The RL environment.

    Returns:
      Per-environment cumulative work in joules. Shape: ``(B,)``.
      Returns zeros if no electrical motors are present.
    """
    # Initialize on first call
    if self._cumulative_work is None:
      self._num_envs = env.num_envs
      self._device = env.device
      self._dt = env.physics_dt * env.cfg.decimation
      self._cumulative_work = torch.zeros(
        self._num_envs, device=self._device, dtype=torch.float32
      )

    entity = env.scene.entities.get(self._entity_name)
    if entity is None:
      return self._cumulative_work

    from mjlab.actuator import ElectricalMotorActuator

    # Accumulate work from all electrical motors: W += τ * ω * dt
    for actuator in entity._actuators:
      if isinstance(actuator, ElectricalMotorActuator):
        # Get joint velocities for this actuator's joints
        joint_indices = actuator._target_indices
        joint_velocities = entity._data.joint_vel[:, joint_indices]  # (B, N)

        # Estimate torque from motor model: τ = Kt * I * N
        # (This is an approximation - actual applied torque may differ due to dynamics)
        torque_est = (
          actuator.motor_spec.motor_constant_kt
          * actuator.current
          * actuator.motor_spec.gear_ratio
        )  # (B, N)

        # Power = τ * ω
        power = torque_est * joint_velocities  # (B, N)

        # Work increment: W += P * dt
        work_increment = torch.sum(power, dim=1) * self._dt  # (B,)
        self._cumulative_work += work_increment

    return self._cumulative_work

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    """Reset cumulative work for specified environments.

    Args:
      env_ids: Environment indices to reset. If None, resets all.

    Returns:
      Dictionary with final work values before reset (for logging).
    """
    if self._cumulative_work is None:
      return {}

    if env_ids is None:
      final_values = {
        f"final_work_j_env_{i}": self._cumulative_work[i].item()
        for i in range(self._num_envs)
      }
      self._cumulative_work.zero_()
    else:
      final_values = {
        f"final_work_j_env_{i}": self._cumulative_work[i].item()
        for i in env_ids.tolist()
      }
      self._cumulative_work[env_ids] = 0.0

    return final_values


# Convenience Helper


def electrical_metrics_preset(
  include_motor: bool = True,
  include_battery: bool = True,
  entity_name: str = "robot",
) -> dict[str, "MetricsTermCfg"]:
  """Returns standard electrical metrics configuration.

  Provides a convenient preset of motor and battery metrics for real-time
  visualization in the Viser viewer.

  Args:
    include_motor: Whether to include motor metrics (current, voltage, power, temperature, back-EMF).
    include_battery: Whether to include battery metrics (SOC, voltage, current, power, temperature).
    entity_name: Name of the entity containing electrical motors.

  Returns:
    Dictionary of MetricsTermCfg suitable for ``env_cfg.metrics``.

  Example:
    Basic usage::

      from mjlab.envs.mdp.metrics import electrical_metrics_preset

      cfg.metrics = electrical_metrics_preset()

    Combine with custom metrics::

      cfg.metrics = {
          **electrical_metrics_preset(),
          "my_metric": MetricsTermCfg(func=my_func),
      }

    Only battery metrics::

      cfg.metrics = electrical_metrics_preset(include_motor=False)
  """
  from mjlab.managers import MetricsTermCfg

  metrics = {}

  if include_motor:
    metrics.update(
      {
        "motor_current_avg": MetricsTermCfg(
          func=motor_current_avg,
          params={"entity_name": entity_name},
        ),
        "motor_voltage_avg": MetricsTermCfg(
          func=motor_voltage_avg,
          params={"entity_name": entity_name},
        ),
        "motor_power_total": MetricsTermCfg(
          func=motor_power_total,
          params={"entity_name": entity_name},
        ),
        "motor_temperature_max": MetricsTermCfg(
          func=motor_temperature_max,
          params={"entity_name": entity_name},
        ),
        "motor_back_emf_avg": MetricsTermCfg(
          func=motor_back_emf_avg,
          params={"entity_name": entity_name},
        ),
      }
    )

  if include_battery:
    metrics.update(
      {
        "battery_soc": MetricsTermCfg(func=battery_soc),
        "battery_voltage": MetricsTermCfg(func=battery_voltage),
        "battery_current": MetricsTermCfg(func=battery_current),
        "battery_power": MetricsTermCfg(func=battery_power),
        "battery_temperature": MetricsTermCfg(func=battery_temperature),
      }
    )

  return metrics
