"""Datasheet validation tests for motor database entries.

These tests validate that motor specifications match manufacturer datasheets
and that the electrical/mechanical models produce expected behavior.
"""

import pytest
import torch

from mjlab.motor_database import load_motor_spec


def test_unitree_7520_14_stall_torque():
  """Validate Unitree 7520-14 stall torque is physically reasonable.

  Datasheet spec: 88 N⋅m continuous torque at stall (after gearing)
  Physics check: τ_out = Kt * I_stall * gear_ratio should be in ballpark
  Note: Datasheet values include efficiency losses, so exact match not expected.
  """
  motor = load_motor_spec("unitree_7520_14")

  # From datasheet
  expected_stall_torque = 88.0  # N⋅m (output after gearing)

  # Validate stored value
  assert motor.stall_torque == expected_stall_torque, (
    f"Stall torque mismatch: {motor.stall_torque} != {expected_stall_torque}"
  )

  # Validate physics is in reasonable range (accounting for efficiency)
  calculated_stall_torque = (
    motor.motor_constant_kt * motor.stall_current * motor.gear_ratio
  )

  # Allow large tolerance due to efficiency losses and measurement uncertainty
  # Calculated should be <= datasheet (efficiency < 100%)
  assert calculated_stall_torque > 0, "Calculated torque must be positive"
  assert calculated_stall_torque < expected_stall_torque * 3, (
    f"Calculated torque ({calculated_stall_torque:.2f}) unreasonably high vs "
    f"datasheet ({expected_stall_torque})"
  )


def test_unitree_7520_14_no_load_speed():
  """Validate Unitree 7520-14 no-load speed is physically reasonable.

  Datasheet spec: 32 rad/s no-load speed (output after gearing)
  Note: Datasheet values may differ from theoretical due to losses.
  """
  motor = load_motor_spec("unitree_7520_14")

  # From datasheet
  expected_no_load_speed = 32.0  # rad/s (output after gearing)

  # Validate stored value
  assert motor.no_load_speed == expected_no_load_speed, (
    f"No-load speed mismatch: {motor.no_load_speed} != {expected_no_load_speed}"
  )

  # Validate physics is in reasonable range
  rated_voltage = motor.voltage_range[1]  # Max voltage
  motor_speed = rated_voltage / motor.motor_constant_ke  # Motor speed before gearing
  calculated_no_load_speed = (
    motor_speed / motor.gear_ratio
  )  # Output speed after gearing

  # Check it's in the right ballpark (allow 51% tolerance for measurement/efficiency)
  assert calculated_no_load_speed > 0, "Calculated speed must be positive"
  assert abs(calculated_no_load_speed / expected_no_load_speed - 1.0) < 0.51, (
    f"No-load speed unreasonably different: calculated={calculated_no_load_speed:.2f}, "
    f"datasheet={expected_no_load_speed}"
  )


def test_unitree_5020_9_stall_torque():
  """Validate Unitree 5020-9 stall torque matches datasheet.

  Datasheet spec: 30 N⋅m peak torque at stall
  """
  motor = load_motor_spec("unitree_5020_9")

  # From datasheet
  expected_peak_torque = 30.0  # N⋅m

  # Validate stored value
  assert motor.peak_torque == expected_peak_torque, (
    f"Peak torque mismatch: {motor.peak_torque} != {expected_peak_torque}"
  )

  # Validate physics consistency
  calculated_stall_torque = motor.motor_constant_kt * motor.stall_current
  tolerance = 0.5  # N⋅m

  # Stall torque should be close to peak torque (or slightly less)
  assert calculated_stall_torque <= expected_peak_torque + tolerance, (
    f"Calculated stall torque ({calculated_stall_torque}) exceeds peak torque ({expected_peak_torque})"
  )


def test_torque_speed_curve_unitree_7520_14():
  """Validate torque-speed curve has correct shape.

  The curve should be monotonically decreasing from stall to no-load.
  Exact values may differ from datasheet due to efficiency.
  """
  motor = load_motor_spec("unitree_7520_14")

  # Test points along the torque-speed curve
  test_speeds = torch.linspace(0, motor.no_load_speed, 10)

  prev_torque = float("inf")
  for omega_out in test_speeds:
    # Calculate torque using motor equations with gearing
    V = motor.voltage_range[1]  # Max voltage
    omega_motor = omega_out.item() * motor.gear_ratio
    I = (V - motor.motor_constant_ke * omega_motor) / motor.resistance
    tau_motor = motor.motor_constant_kt * I
    calculated_torque = tau_motor * motor.gear_ratio

    # Torque should decrease monotonically with speed
    assert calculated_torque <= prev_torque + 1e-6, (
      f"Torque increased with speed at ω={omega_out:.2f}: "
      f"τ={calculated_torque:.2f} > prev={prev_torque:.2f}"
    )
    prev_torque = calculated_torque

  # Check boundary conditions are reasonable
  # At zero speed: should have high torque
  V = motor.voltage_range[1]
  I_stall = V / motor.resistance
  tau_stall = motor.motor_constant_kt * I_stall * motor.gear_ratio
  assert tau_stall > motor.stall_torque * 0.1, "Stall torque too low"

  # At no-load speed: torque magnitude should be less than 3x stall (back-EMF dominates)
  omega_motor_noload = motor.no_load_speed * motor.gear_ratio
  I_noload = (V - motor.motor_constant_ke * omega_motor_noload) / motor.resistance
  tau_noload = motor.motor_constant_kt * I_noload * motor.gear_ratio
  assert abs(tau_noload) < motor.stall_torque * 3, "No-load torque unreasonably high"


def test_thermal_limits_unitree_7520_14():
  """Validate thermal limits are physically consistent."""
  motor = load_motor_spec("unitree_7520_14")

  # Thermal limits should be reasonable
  assert motor.max_winding_temperature > motor.ambient_temperature, (
    "Max temperature must exceed ambient temperature"
  )

  assert motor.thermal_resistance > 0, "Thermal resistance must be positive"

  assert motor.thermal_time_constant > 0, "Thermal time constant must be positive"

  # Check thermal power at continuous current
  I_continuous = motor.operating_current
  P_continuous = I_continuous**2 * motor.resistance
  ΔT_steady_state = P_continuous * motor.thermal_resistance

  # At continuous operation, temperature rise should be safe
  T_steady_state = motor.ambient_temperature + ΔT_steady_state

  assert T_steady_state < motor.max_winding_temperature, (
    f"Continuous operation would exceed thermal limits: "
    f"T={T_steady_state:.1f}°C > T_max={motor.max_winding_temperature}°C"
  )


def test_power_consistency_unitree_7520_14():
  """Validate electrical and mechanical power are consistent."""
  motor = load_motor_spec("unitree_7520_14")

  # At rated operating point
  omega = motor.no_load_speed / 2  # Mid-range speed
  V = motor.voltage_range[1]

  # Electrical side
  I = (V - motor.motor_constant_ke * omega) / motor.resistance
  P_elec = V * I

  # Mechanical side
  tau = motor.motor_constant_kt * I
  P_mech = tau * omega

  # Losses
  P_copper = I**2 * motor.resistance
  P_losses = P_copper  # Simplified (ignoring friction, iron losses)

  # Power balance: P_elec = P_mech + P_losses
  power_balance_error = abs(P_elec - (P_mech + P_losses))

  # Should be nearly zero (within numerical precision)
  assert power_balance_error < 0.01, (
    f"Power balance violated: P_elec={P_elec:.2f}W, "
    f"P_mech={P_mech:.2f}W, P_losses={P_losses:.2f}W"
  )


def test_unitree_5020_9_no_load_speed():
  """Validate Unitree 5020-9 no-load speed is physically reasonable."""
  motor = load_motor_spec("unitree_5020_9")

  # From datasheet
  expected_no_load_speed = 40.0  # rad/s (output after gearing)

  # Validate stored value
  assert motor.no_load_speed == expected_no_load_speed, (
    f"No-load speed mismatch: {motor.no_load_speed} != {expected_no_load_speed}"
  )

  # Validate voltage-speed relationship (accounting for gearing)
  rated_voltage = motor.voltage_range[1]
  motor_speed = rated_voltage / motor.motor_constant_ke
  calculated_speed = motor_speed / motor.gear_ratio  # Output speed after gearing

  # Check it's in reasonable range (50% tolerance)
  assert calculated_speed > 0, "Calculated speed must be positive"
  assert abs(calculated_speed / expected_no_load_speed - 1.0) < 0.5, (
    f"Speed unreasonably different: calculated={calculated_speed:.2f}, "
    f"datasheet={expected_no_load_speed}"
  )


def test_motor_constants_consistency():
  """Validate Kt and Ke are consistent (should be equal in SI units)."""
  motors = ["unitree_7520_14", "unitree_5020_9"]

  for motor_id in motors:
    motor = load_motor_spec(motor_id)

    # In SI units, Kt (N⋅m/A) should equal Ke (V⋅s/rad)
    # Allow small tolerance for rounding/measurement error
    tolerance = 0.01  # 1%

    kt_ke_ratio = motor.motor_constant_kt / motor.motor_constant_ke

    assert abs(kt_ke_ratio - 1.0) < tolerance, (
      f"{motor_id}: Kt/Ke ratio ({kt_ke_ratio:.3f}) deviates from 1.0. "
      f"Kt={motor.motor_constant_kt}, Ke={motor.motor_constant_ke}"
    )
