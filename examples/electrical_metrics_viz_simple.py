"""
Simple visualization demo for Phase 6 electrical metrics.

This script creates a pendulum robot with an electrical motor and battery,
then runs a simple simulation while displaying real-time electrical metrics
in the terminal.

For Viser web-based visualization, you would need to integrate with ManagerBasedRlEnv.
This demo shows the metrics updating in the terminal as a simpler alternative.

Run with:
    uv run python examples/electrical_metrics_viz_simple.py
    uv run python examples/electrical_metrics_viz_simple.py --steps 100
"""

from __future__ import annotations

import argparse
import time

import mujoco
import torch

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.battery import BatteryManagerCfg
from mjlab.battery_database import load_battery_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.motor_database import load_motor_spec
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg


def create_pendulum_xml() -> str:
  """Create a simple pendulum XML for visualization."""
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


def main():
  parser = argparse.ArgumentParser(
    description="Simple electrical metrics visualization"
  )
  parser.add_argument(
    "--steps", type=int, default=200, help="Number of simulation steps"
  )
  parser.add_argument(
    "--num_envs", type=int, default=2, help="Number of parallel environments"
  )
  args = parser.parse_args()

  print("=" * 70)
  print("Electrical Metrics Visualization Demo")
  print("=" * 70)

  # Setup
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"\nDevice: {device}")
  print(f"Environments: {args.num_envs}")
  print(f"Steps: {args.steps}")

  # Load specs
  motor_spec = load_motor_spec("unitree_7520_14")
  battery_spec = load_battery_spec("unitree_g1_9ah")

  print(f"\nMotor: {motor_spec.manufacturer} {motor_spec.model}")
  print(f"Motor torque: {motor_spec.continuous_torque} N⋅m")
  print(f"Battery: {battery_spec.manufacturer} {battery_spec.model}")
  print(f"Battery capacity: {battery_spec.capacity_ah} Ah")

  # Create scene with electrical motor and battery
  scene_cfg = SceneCfg(
    num_envs=args.num_envs,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(create_pendulum_xml()),
        articulation=EntityArticulationInfoCfg(
          actuators=(
            ElectricalMotorActuatorCfg(
              target_names_expr=("hinge",),
              motor_spec=motor_spec,
              stiffness=200.0,
              damping=10.0,
              saturation_effort=motor_spec.peak_torque,
              velocity_limit=motor_spec.no_load_speed,
              effort_limit=motor_spec.continuous_torque,
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

  print("\nInitializing simulation...")
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(
    num_envs=args.num_envs,
    cfg=SimulationCfg(),
    model=model,
    device=device,
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot = scene.entities["robot"]
  battery = scene._battery_manager
  actuator = robot._actuators[0]

  print("✓ Simulation initialized")
  print("\nRunning simulation with electrical metrics visualization...\n")
  print("-" * 70)
  print(
    f"{'Step':<6} {'SOC(%)':<8} {'Battery(V)':<12} {'Current(A)':<12} {'Power(W)':<10} {'Temp(°C)':<8}"
  )
  print("-" * 70)

  # Simulation loop with simple PD control
  dt = 0.002
  target_angle = 0.5  # Target angle in radians
  kp = 100.0
  kd = 10.0

  for step in range(args.steps):
    # Simple PD control
    current_angle = robot._data.joint_pos[0, 0].item()
    current_vel = robot._data.joint_vel[0, 0].item()

    error = target_angle - current_angle
    torque = kp * error - kd * current_vel
    torque_cmd = torch.tensor([[torque]], device=device).expand(args.num_envs, 1)

    robot.set_joint_effort_target(torque_cmd)

    # Step simulation
    sim.step()
    actuator.update(dt)
    if battery is not None:
      battery.update(dt)

    # Display metrics every 10 steps
    if step % 10 == 0:
      if battery is not None and battery.soc is not None:
        soc = battery.soc[0].item() * 100
        voltage = battery.voltage[0].item() if battery.voltage is not None else 0.0
        current = battery.current[0].item() if battery.current is not None else 0.0
        power = battery.power_out[0].item() if battery.power_out is not None else 0.0
        temp = battery.temperature[0].item() if battery.temperature is not None else 0.0

        print(
          f"{step:<6} {soc:<8.2f} {voltage:<12.2f} {current:<12.2f} {power:<10.1f} {temp:<8.1f}"
        )

    # Small delay for visualization (optional)
    if step % 20 == 0:
      time.sleep(0.05)

  print("-" * 70)
  print("\n✓ Simulation complete!")

  # Final statistics
  if battery is not None and battery.soc is not None:
    final_soc = battery.soc[0].item() * 100
    initial_soc = 100.0
    soc_drop = initial_soc - final_soc

    print(f"\nFinal Statistics:")
    print(f"  Initial SOC: {initial_soc:.1f}%")
    print(f"  Final SOC: {final_soc:.1f}%")
    print(f"  SOC drop: {soc_drop:.3f}%")
    print(
      f"  Battery drained: {soc_drop / 100 * battery_spec.capacity_ah * 1000:.2f} mAh"
    )

  # Motor statistics
  if actuator.current is not None:
    avg_current = torch.mean(torch.abs(actuator.current)).item()
    max_current = torch.max(torch.abs(actuator.current)).item()
    print(f"\nMotor Statistics:")
    print(f"  Average current: {avg_current:.2f} A")
    print(f"  Peak current: {max_current:.2f} A")
    if actuator.winding_temperature is not None:
      max_temp = torch.max(actuator.winding_temperature).item()
      print(f"  Max temperature: {max_temp:.1f}°C")

  print("\n" + "=" * 70)
  print("Phase 6 Electrical Metrics Demo Complete!")
  print("=" * 70)
  print("\nTo see metrics in Viser web viewer:")
  print("  1. Create a ManagerBasedRlEnv with electrical_metrics_preset()")
  print("  2. Run with Viser viewer enabled")
  print("  3. Open browser to see real-time plots in Metrics tab")
  print("\nSee examples/electrical_metrics_demo.py for usage examples.")


if __name__ == "__main__":
  main()
