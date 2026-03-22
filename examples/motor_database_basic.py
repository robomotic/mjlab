"""
Example: Using the Motor Database

This example demonstrates how to load and use motor specifications
from the motor database in mjlab.
"""

from mjlab.motor_database import (
  MotorSpecification,
  get_default_search_paths,
  load_motor_spec,
)


def main():
  print("=" * 70)
  print("Motor Database Example")
  print("=" * 70)

  # Example 1: Load built-in motor by ID
  print("\n1. Loading built-in motor by ID:")
  hip_motor = load_motor_spec("unitree_7520_14")
  print(f"   Motor ID: {hip_motor.motor_id}")
  print(f"   Manufacturer: {hip_motor.manufacturer} {hip_motor.model}")
  print(f"   Continuous torque: {hip_motor.continuous_torque} N⋅m")
  print(f"   Gear ratio: {hip_motor.gear_ratio}")
  print(f"   Resistance: {hip_motor.resistance} Ω")
  print(f"   Motor constant (Kt): {hip_motor.motor_constant_kt} N⋅m/A")

  # Example 2: Load another motor
  print("\n2. Loading ankle motor:")
  ankle_motor = load_motor_spec("unitree_5020_9")
  print(f"   Motor ID: {ankle_motor.motor_id}")
  print(f"   Continuous torque: {ankle_motor.continuous_torque} N⋅m")
  print(f"   No-load speed: {ankle_motor.no_load_speed} rad/s")

  # Example 3: Compare motors
  print("\n3. Comparing motors:")
  print(f"   Hip motor torque:   {hip_motor.continuous_torque:.1f} N⋅m")
  print(f"   Ankle motor torque: {ankle_motor.continuous_torque:.1f} N⋅m")
  print(
    f"   Torque ratio: {hip_motor.continuous_torque / ankle_motor.continuous_torque:.2f}x"
  )

  # Example 4: Check electrical properties
  print("\n4. Electrical properties:")
  print("   Hip motor:")
  print(
    f"     Voltage range: {hip_motor.voltage_range[0]}-{hip_motor.voltage_range[1]} V"
  )
  print(f"     Stall current: {hip_motor.stall_current} A")
  print(f"     Resistance: {hip_motor.resistance} Ω")
  print(f"     Inductance: {hip_motor.inductance * 1000:.2f} mH")

  # Example 5: Thermal properties
  print("\n5. Thermal properties:")
  print(f"   Max temperature: {hip_motor.max_winding_temperature}°C")
  print(f"   Thermal resistance: {hip_motor.thermal_resistance} °C/W")
  print(f"   Thermal time constant: {hip_motor.thermal_time_constant} s")

  # Example 6: Show search paths
  print("\n6. Motor database search paths:")
  paths = get_default_search_paths()
  for i, path in enumerate(paths, 1):
    print(f"   {i}. {path}")

  # Example 7: Create a custom motor spec programmatically
  print("\n7. Creating custom motor spec:")
  custom_motor = MotorSpecification(
    motor_id="custom_motor_example",
    manufacturer="Example Corp",
    model="EX-100",
    gear_ratio=20.0,
    continuous_torque=50.0,
    resistance=0.5,
    inductance=0.0002,
    motor_constant_kt=0.12,
    motor_constant_ke=0.12,
  )
  print(f"   Created: {custom_motor.motor_id}")
  print(f"   Torque: {custom_motor.continuous_torque} N⋅m")
  print(f"   Gear ratio: {custom_motor.gear_ratio}")

  print("\n" + "=" * 70)
  print("✅ Motor database example complete!")
  print("=" * 70)
  print("\nNext: Use these motor specs with ElectricalMotorActuator (Phase 2)")


if __name__ == "__main__":
  main()
