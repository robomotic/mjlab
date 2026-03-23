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
