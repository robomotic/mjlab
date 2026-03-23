#!/usr/bin/env python3
"""
Quick verification script for motor database implementation.
This can be run without the full mjlab environment.
"""

import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the motor database modules (this won't trigger mjlab.__init__)
from mjlab.motor_database.database import (
  BUILTIN_MOTORS_PATH,
  get_default_search_paths,
  load_motor_spec,
)
from mjlab.motor_database.motor_spec import MotorSpecification

print("=" * 60)
print("Motor Database Verification")
print("=" * 60)

print("\n1. Testing motor spec creation...")
motor = MotorSpecification(
  motor_id="verification_motor",
  manufacturer="VerifyTest",
  model="VT-1",
  gear_ratio=10.0,
)
print(f"   ✓ Created motor: {motor.motor_id}")
print(f"   ✓ Gear ratio: {motor.gear_ratio}")
print(f"   ✓ Default sensors: {motor.feedback_sensors}")

print("\n2. Loading built-in motors...")
motors_to_test = ["unitree_7520_14", "unitree_5020_9", "test_motor"]
for motor_id in motors_to_test:
  m = load_motor_spec(motor_id)
  print(
    f"   ✓ {motor_id}: {m.manufacturer} {m.model}, torque={m.continuous_torque} N⋅m"
  )

print("\n3. Testing search paths...")
paths = get_default_search_paths()
print(f"   ✓ Found {len(paths)} search path(s)")
print(f"   ✓ Built-in path included: {BUILTIN_MOTORS_PATH in paths}")

print("\n4. Testing file loading...")
test_file = BUILTIN_MOTORS_PATH / "unitree_7520_14.json"
motor_from_file = load_motor_spec(file=test_file)
print(f"   ✓ Loaded from file: {motor_from_file.motor_id}")

print("\n5. Testing error handling...")
try:
  load_motor_spec("nonexistent_motor_xyz")
  print("   ✗ Should have raised FileNotFoundError")
  sys.exit(1)
except FileNotFoundError:
  print("   ✓ FileNotFoundError raised correctly")

try:
  load_motor_spec()  # No arguments
  print("   ✗ Should have raised ValueError")
  sys.exit(1)
except ValueError:
  print("   ✓ ValueError raised correctly")

print("\n" + "=" * 60)
print("✅ All verification tests passed!")
print("=" * 60)
print("\nMotor database is ready for use.")
print("Next steps: Implement ElectricalMotorActuator (Phase 2)")
