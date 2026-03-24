"""Tests for cable-powered (infinite power) electrical motor mode."""

from __future__ import annotations

import mujoco
import torch

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.metrics import electrical_metrics_preset
from mjlab.motor_database import load_motor_spec
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.velocity.config.g1.env_cfgs_electric import (
  unitree_g1_flat_electric_cable_env_cfg,
)


def create_test_pendulum_xml() -> str:
  """Create a simple pendulum XML for testing."""
  return """
  <mujoco model="pendulum">
    <option timestep="0.002" integrator="implicitfast"/>
    <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
      <body name="pole" pos="0 0 2">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -1" rgba="0 .9 0 1"/>
        <geom type="sphere" size="0.1" pos="0 0 -1" rgba="0 .9 0 1" mass="1.0"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" gainprm="1" biasprm="0 -1 0" dyntype="none"/>
    </actuator>
  </mujoco>
  """


def test_electrical_motor_no_battery_full_voltage():
  """Motors without battery should have full rated voltage."""
  # Create scene with electrical motors but NO battery
  motor_spec = load_motor_spec("unitree_7520_14")

  scene_cfg = SceneCfg(
    num_envs=4,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(create_test_pendulum_xml()),
        articulation=EntityArticulationInfoCfg(
          actuators=(
            ElectricalMotorActuatorCfg(
              target_names_expr=("hinge",),
              motor_spec=motor_spec,
              stiffness=200.0,
              damping=10.0,
              saturation_effort=motor_spec.peak_torque,
              velocity_limit=motor_spec.no_load_speed,
            ),
          )
        ),
      )
    },
    # battery=None  # No battery → cable-powered (default)
  )

  scene = Scene(scene_cfg, device="cpu")
  model = scene.compile()
  sim = Simulation(
    num_envs=4,
    cfg=SimulationCfg(),
    model=model,
    device="cpu",
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)

  # Verify actuator has full rated voltage from motor spec
  actuator = scene.entities["robot"]._actuators[0]
  expected_voltage = motor_spec.voltage_range[1]  # 24.0V for unitree_7520_14
  assert torch.allclose(actuator._voltage_max, torch.full((4, 1), expected_voltage))

  # Verify no battery manager was created
  assert scene._battery_manager is None

  # Step simulation multiple times with control commands
  robot = scene.entities["robot"]
  for _ in range(100):
    # Apply torque command
    torque_cmd = torch.tensor([[10.0]], device="cpu").expand(4, 1)
    robot.set_joint_effort_target(torque_cmd)

    # Write to sim, step physics, update actuators
    scene.write_data_to_sim()
    sim.step()
    actuator.update(dt=0.02)

  # Voltage should remain at rated maximum (no sag from battery)
  assert torch.allclose(actuator._voltage_max, torch.full((4, 1), expected_voltage))


def test_electrical_metrics_preset_no_battery():
  """electrical_metrics_preset with include_battery=False should work."""
  metrics = electrical_metrics_preset(
    include_motor=True,
    include_battery=False,
  )

  # Should have motor metrics
  assert "motor_current_avg" in metrics
  assert "motor_voltage_avg" in metrics
  assert "motor_power_total" in metrics
  assert "motor_temperature_max" in metrics
  assert "motor_back_emf_avg" in metrics

  # Should NOT have battery metrics
  assert "battery_soc" not in metrics
  assert "battery_voltage" not in metrics
  assert "battery_current" not in metrics
  assert "battery_power" not in metrics
  assert "battery_temperature" not in metrics


def test_cable_powered_env_cfg():
  """G1 cable-powered environment should initialize without battery."""
  cfg = unitree_g1_flat_electric_cable_env_cfg(play=False)

  # Should have electrical motors configured
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  assert len(robot_cfg.articulation.actuators) == 2  # Hip/knee + ankle/arm actuators

  # Verify actuators are ElectricalMotorActuatorCfg
  for actuator_cfg in robot_cfg.articulation.actuators:
    assert isinstance(actuator_cfg, ElectricalMotorActuatorCfg)
    assert actuator_cfg.motor_spec is not None

  # Should NOT have battery configured
  assert cfg.scene.battery is None

  # Should have motor metrics but not battery metrics
  assert "motor_current_avg" in cfg.metrics
  assert "motor_voltage_avg" in cfg.metrics
  assert "motor_power_total" in cfg.metrics

  # Battery metrics should be excluded
  assert "battery_soc" not in cfg.metrics
  assert "battery_voltage" not in cfg.metrics
  assert "battery_current" not in cfg.metrics
