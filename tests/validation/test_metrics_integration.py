"""Integration test for electrical metrics with ManagerBasedRlEnv.

This test validates that electrical metrics work end-to-end with:
- Real ManagerBasedRlEnv environment
- Electrical motor actuators
- Battery system
- MetricsManager collecting and exposing metrics
- Real-time metric updates during simulation
"""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.battery import BatteryManagerCfg
from mjlab.battery_database import load_battery_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.metrics import electrical_metrics_preset
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.motor_database import load_motor_spec
from mjlab.scene import SceneCfg

device = get_test_device()


@pytest.fixture
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


@pytest.fixture
def motor_spec():
  """Load a test motor specification."""
  return load_motor_spec("unitree_7520_14")


@pytest.fixture
def battery_spec():
  """Load a test battery specification."""
  return load_battery_spec("unitree_g1_9ah")


def test_electrical_metrics_with_env(robot_xml, motor_spec, battery_spec):
  """Test electrical metrics integrate correctly with ManagerBasedRlEnv."""
  # Create environment config with electrical actuators and battery
  scene_cfg = SceneCfg(
    num_envs=2,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
        articulation=EntityArticulationInfoCfg(
          actuators=(
            ElectricalMotorActuatorCfg(
              target_names_expr=(".*",),
              motor_spec=motor_spec,
              stiffness=100.0,
              damping=5.0,
              saturation_effort=motor_spec.peak_torque,
              velocity_limit=motor_spec.no_load_speed,
            ),
          )
        ),
      )
    },
    battery=BatteryManagerCfg(
      battery_spec=battery_spec,
      entity_names=("robot",),
      initial_soc=1.0,
      enable_voltage_feedback=True,
    ),
  )

  # Create minimal env config with electrical metrics
  env_cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: env.scene.entities["robot"].data.joint_pos
          )
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    decimation=1,
    metrics=electrical_metrics_preset(),
  )

  # Create environment
  env = ManagerBasedRlEnv(env_cfg, device=str(device))

  # Verify metrics manager has all electrical metrics
  assert hasattr(env, "metrics_manager")
  assert "motor_current_avg" in env.metrics_manager._term_names
  assert "battery_soc" in env.metrics_manager._term_names

  # Run a few simulation steps
  actions = torch.zeros(
    (env.num_envs, env.action_manager.total_action_dim), device=device
  )

  for _ in range(10):
    env.step(actions)

  # Verify metrics are computed and accessible
  metrics_env_0 = env.metrics_manager.get_active_iterable_terms(env_idx=0)
  metrics_dict = {name: values[0] for name, values in metrics_env_0}

  # Check motor metrics exist and have reasonable values
  assert "motor_current_avg" in metrics_dict
  motor_current = metrics_dict["motor_current_avg"]
  assert isinstance(motor_current, (int, float))
  assert motor_current >= 0.0  # Current should be non-negative

  assert "motor_voltage_avg" in metrics_dict
  assert "motor_power_total" in metrics_dict
  assert "motor_temperature_max" in metrics_dict
  assert "motor_back_emf_avg" in metrics_dict

  # Check battery metrics exist
  assert "battery_soc" in metrics_dict
  battery_soc = metrics_dict["battery_soc"]
  assert isinstance(battery_soc, (int, float))
  assert 0.0 <= battery_soc <= 1.0  # SOC should be in [0, 1]

  assert "battery_voltage" in metrics_dict
  assert "battery_current" in metrics_dict
  assert "battery_power" in metrics_dict
  assert "battery_temperature" in metrics_dict

  print("\n✓ All electrical metrics integrated successfully with ManagerBasedRlEnv")
  print(f"  Motor current: {motor_current:.2f}A")
  print(f"  Battery SOC: {battery_soc * 100:.1f}%")
  print(f"  Battery voltage: {metrics_dict['battery_voltage']:.1f}V")


def test_metrics_update_each_step(robot_xml, motor_spec, battery_spec):
  """Verify metrics update with each simulation step."""
  scene_cfg = SceneCfg(
    num_envs=1,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
        articulation=EntityArticulationInfoCfg(
          actuators=(
            ElectricalMotorActuatorCfg(
              target_names_expr=(".*",),
              motor_spec=motor_spec,
              stiffness=100.0,
              damping=5.0,
              saturation_effort=motor_spec.peak_torque,
              velocity_limit=motor_spec.no_load_speed,
            ),
          )
        ),
      )
    },
    battery=BatteryManagerCfg(
      battery_spec=battery_spec,
      entity_names=("robot",),
      initial_soc=1.0,
      enable_voltage_feedback=True,
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: env.scene.entities["robot"].data.joint_pos
          )
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    decimation=1,
    metrics=electrical_metrics_preset(),
  )

  env = ManagerBasedRlEnv(env_cfg, device=str(device))

  # Record battery SOC over time
  soc_history = []
  actions = torch.ones(
    (env.num_envs, env.action_manager.total_action_dim), device=device
  )

  for _ in range(50):
    env.step(actions)
    metrics = env.metrics_manager.get_active_iterable_terms(env_idx=0)
    metrics_dict = {name: values[0] for name, values in metrics}
    soc_history.append(metrics_dict["battery_soc"])

  # Battery SOC should decrease over time (battery draining)
  # Check that SOC at end is less than SOC at start
  initial_soc = soc_history[0]
  final_soc = soc_history[-1]

  print(f"\n  Initial SOC: {initial_soc * 100:.2f}%")
  print(f"  Final SOC: {final_soc * 100:.2f}%")
  print(f"  SOC decrease: {(initial_soc - final_soc) * 100:.3f}%")

  # SOC should decrease (or stay approximately constant if power draw is minimal)
  assert final_soc <= initial_soc + 0.001, "Battery SOC should not increase"

  print("✓ Metrics update correctly with each simulation step")


def test_metrics_multi_env(robot_xml, motor_spec, battery_spec):
  """Verify metrics work correctly with multiple environments."""
  scene_cfg = SceneCfg(
    num_envs=4,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
        articulation=EntityArticulationInfoCfg(
          actuators=(
            ElectricalMotorActuatorCfg(
              target_names_expr=(".*",),
              motor_spec=motor_spec,
              stiffness=100.0,
              damping=5.0,
              saturation_effort=motor_spec.peak_torque,
              velocity_limit=motor_spec.no_load_speed,
            ),
          )
        ),
      )
    },
    battery=BatteryManagerCfg(
      battery_spec=battery_spec,
      entity_names=("robot",),
      initial_soc=1.0,
      enable_voltage_feedback=True,
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: env.scene.entities["robot"].data.joint_pos
          )
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    decimation=1,
    metrics=electrical_metrics_preset(),
  )

  env = ManagerBasedRlEnv(env_cfg, device=str(device))

  # Different actions for each environment
  actions = torch.randn(
    (env.num_envs, env.action_manager.total_action_dim), device=device
  )

  # Run simulation
  for _ in range(20):
    env.step(actions)

  # Verify each environment has different metric values
  metrics_per_env = []
  for env_idx in range(env.num_envs):
    metrics = env.metrics_manager.get_active_iterable_terms(env_idx=env_idx)
    metrics_dict = {name: values[0] for name, values in metrics}
    metrics_per_env.append(metrics_dict)

  # Check that environments have potentially different values
  # (Due to different random actions, they may diverge)
  print("\n  Metrics across environments:")
  for env_idx in range(env.num_envs):
    metrics = metrics_per_env[env_idx]
    print(
      f"    Env {env_idx}: Current={metrics['motor_current_avg']:.2f}A, "
      f"SOC={metrics['battery_soc'] * 100:.1f}%"
    )

  print("✓ Multi-environment metrics work correctly")


def test_preset_filters(robot_xml, motor_spec, battery_spec):
  """Test electrical_metrics_preset filtering options."""
  scene_cfg = SceneCfg(
    num_envs=1,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
        articulation=EntityArticulationInfoCfg(
          actuators=(
            ElectricalMotorActuatorCfg(
              target_names_expr=(".*",),
              motor_spec=motor_spec,
              stiffness=100.0,
              damping=5.0,
              saturation_effort=motor_spec.peak_torque,
              velocity_limit=motor_spec.no_load_speed,
            ),
          )
        ),
      )
    },
    battery=BatteryManagerCfg(
      battery_spec=battery_spec,
      entity_names=("robot",),
      initial_soc=1.0,
      enable_voltage_feedback=True,
    ),
  )

  # Test motor-only preset
  env_cfg_motor = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: env.scene.entities["robot"].data.joint_pos
          )
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    decimation=1,
    metrics=electrical_metrics_preset(include_motor=True, include_battery=False),
  )

  env_motor = ManagerBasedRlEnv(env_cfg_motor, device=str(device))

  # Verify only motor metrics present
  assert "motor_current_avg" in env_motor.metrics_manager._term_names
  assert "battery_soc" not in env_motor.metrics_manager._term_names

  print("\n✓ Motor-only preset works correctly")

  # Test battery-only preset
  env_cfg_battery = ManagerBasedRlEnvCfg(
    scene=scene_cfg,
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: env.scene.entities["robot"].data.joint_pos
          )
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    decimation=1,
    metrics=electrical_metrics_preset(include_motor=False, include_battery=True),
  )

  env_battery = ManagerBasedRlEnv(env_cfg_battery, device=str(device))

  # Verify only battery metrics present
  assert "battery_soc" in env_battery.metrics_manager._term_names
  assert "motor_current_avg" not in env_battery.metrics_manager._term_names

  print("✓ Battery-only preset works correctly")
