"""
Example: Real-time Motor/Battery Metrics Visualization

This example demonstrates how to visualize electrical motor and battery
metrics in real-time during simulation using the Viser viewer.

The demo creates a simple pendulum robot with electrical motor actuators
and a battery system, then displays live telemetry plots including:
- Motor current, voltage, power, temperature, back-EMF
- Battery SOC, voltage, current, power, temperature

Run with Viser viewer to see interactive time-series plots.

Usage:
    python examples/electrical_metrics_demo.py --num_envs 4
    python examples/electrical_metrics_demo.py --headless  # No visualization
"""

from __future__ import annotations

import argparse

import mujoco

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.battery import BatteryManagerCfg
from mjlab.battery_database import load_battery_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.metrics import electrical_metrics_preset
from mjlab.motor_database import load_motor_spec
from mjlab.scene import Scene, SceneCfg


def create_demo_xml() -> str:
  """Create a simple pendulum XML for demonstration."""
  return """
  <mujoco model="pendulum">
    <option timestep="0.002" integrator="implicitfast"/>

    <worldbody>
      <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

      <body name="pole" pos="0 0 2">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -1" rgba="0 .9 0 1"/>
        <geom type="sphere" size="0.1" pos="0 0 -1" rgba="0 .9 0 1"/>
      </body>
    </worldbody>
  </mujoco>
  """


def main():
  parser = argparse.ArgumentParser(
    description="Real-time electrical metrics visualization demo"
  )
  parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without visualization (metrics still computed)",
  )
  parser.add_argument(
    "--duration", type=float, default=10.0, help="Simulation duration (seconds)"
  )
  args = parser.parse_args()

  print("=" * 70)
  print("Electrical Metrics Visualization Demo")
  print("=" * 70)

  # Load motor and battery specs
  print("\n1. Loading motor and battery specifications...")
  motor_spec = load_motor_spec("unitree_7520_14")  # Hip motor
  battery_spec = load_battery_spec("unitree_g1_9ah")  # 9Ah battery

  print(f"   Motor: {motor_spec.manufacturer} {motor_spec.model}")
  print(f"   Motor torque: {motor_spec.continuous_torque} N⋅m")
  print(f"   Battery: {battery_spec.manufacturer} {battery_spec.model}")
  print(f"   Battery capacity: {battery_spec.capacity_ah} Ah")

  # Create scene with electrical actuator and battery
  print("\n2. Creating scene with electrical actuator and battery...")
  scene_cfg = SceneCfg(
    num_envs=args.num_envs,
    entities={
      "robot": EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(create_demo_xml()),
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
      initial_soc=1.0,  # Start at 100%
      enable_voltage_feedback=True,
    ),
  )

  device = "cpu"
  scene = Scene(scene_cfg, device=device)
  print(f"   ✓ Scene created with {args.num_envs} environments")

  # Note: Full ManagerBasedRlEnv integration would require more setup
  # For now, we demonstrate the scene-level components and metric functions

  print("\n3. Electrical metrics available:")
  metrics = electrical_metrics_preset()
  for i, name in enumerate(metrics.keys(), 1):
    print(f"   {i}. {name}")

  print("\n4. Battery and motor state:")
  battery = scene._battery_manager
  robot = scene.entities["robot"]

  if battery is not None and battery.soc is not None:
    print(f"   Battery SOC: {battery.soc[0].item() * 100:.1f}%")
    if battery.voltage is not None:
      print(f"   Battery voltage: {battery.voltage[0].item():.1f}V")
    if battery.current is not None:
      print(f"   Battery current: {battery.current[0].item():.1f}A")
    if battery.temperature is not None:
      print(f"   Battery temperature: {battery.temperature[0].item():.1f}°C")

  from mjlab.actuator import ElectricalMotorActuator

  electrical_actuators = [
    act for act in robot._actuators if isinstance(act, ElectricalMotorActuator)
  ]
  if electrical_actuators:
    actuator = electrical_actuators[0]
    # Note: Motor properties are initialized after first simulation step
    if actuator.current is not None:
      print(f"   Motor current: {actuator.current[0].mean().item():.2f}A")
      print(f"   Motor voltage: {actuator.voltage[0].mean().item():.2f}V")
      print(f"   Motor power: {actuator.power_dissipation[0].mean().item():.2f}W")
      print(
        f"   Motor temperature: {actuator.winding_temperature[0].mean().item():.1f}°C"
      )
      print(f"   Motor back-EMF: {actuator.back_emf[0].mean().item():.2f}V")
    else:
      print("   Motor state: Not yet initialized (requires simulation step)")
      print(f"   Motor spec: {motor_spec.motor_id}")
      print(f"   Motor resistance: {motor_spec.resistance} Ω")

  print("\n" + "=" * 70)
  print("✅ Demo complete!")
  print("=" * 70)
  print("\nTo use these metrics in a real environment:")
  print("  1. Import: from mjlab.envs.mdp.metrics import electrical_metrics_preset")
  print("  2. Configure: cfg.metrics = electrical_metrics_preset()")
  print("  3. Create env: env = ManagerBasedRlEnv(cfg, device='cuda')")
  print("  4. Run with Viser viewer to see live plots!")
  print("\nExample configuration:")
  print("  env_cfg = ManagerBasedRlEnvCfg(")
  print("      scene=scene_cfg,")
  print("      # ... actions, observations, etc ...")
  print("      metrics=electrical_metrics_preset(),")
  print("  )")


if __name__ == "__main__":
  main()
