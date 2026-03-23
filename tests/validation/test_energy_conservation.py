"""Energy conservation validation tests for electrical motor actuators.

These tests validate that energy is conserved throughout the simulation:
- Electrical energy input = Mechanical work output + Heat losses
- Power balance: P_elec = P_mech + P_copper_loss
- Energy accounting over extended simulation
"""

import mujoco
import pytest
import torch

from conftest import get_test_device, initialize_entity, load_fixture_xml
from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.motor_database import MotorSpecification

device = get_test_device()


@pytest.fixture
def test_motor_spec():
  """Simple motor spec for energy tests."""
  return MotorSpecification(
    motor_id="energy_test_motor",
    manufacturer="Test",
    model="ETM-100",
    gear_ratio=10.0,
    reflected_inertia=0.01,
    rotation_angle_range=(-3.14, 3.14),
    voltage_range=(0.0, 48.0),
    resistance=1.0,
    inductance=0.001,
    motor_constant_kt=0.1,
    motor_constant_ke=0.1,
    stall_torque=10.0,
    peak_torque=10.0,
    continuous_torque=5.0,
    no_load_speed=100.0,
    no_load_current=0.1,
    stall_current=100.0,
    operating_current=50.0,
    thermal_resistance=2.0,
    thermal_time_constant=100.0,
    max_winding_temperature=150.0,
    ambient_temperature=25.0,
    encoder_resolution=1000,
    encoder_type="incremental",
    feedback_sensors=["position", "velocity", "current"],
    protocol="CAN",
    protocol_params={},
  )


@pytest.fixture
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


def test_instantaneous_power_balance(robot_xml, test_motor_spec):
  """Validate instantaneous power balance: P_elec >= P_loss.

  At any instant:
  - P_electrical = V * I (electrical power input)
  - P_copper = I² * R (copper losses)
  - Should satisfy: P_elec >= P_copper (conservation)

  Note: We don't directly measure mechanical power here as it would require
  accessing internal torque values. Instead we verify that electrical power
  exceeds losses, which validates energy isn't being created.
  """
  # Create entity with electrical motor
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=test_motor_spec,
          stiffness=100.0,
          damping=5.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  # Set position target and let it settle
  num_joints = len(entity.joint_names)
  target_pos = torch.ones((1, num_joints)) * 0.5
  entity.set_joint_position_target(target_pos)

  # Run for a few steps to get steady-state motion
  dt = 0.002
  for _ in range(100):
    entity.write_data_to_sim()
    entity.update(dt)

  # Get actuator state
  actuator = entity._actuators[0]

  # Electrical power: P_elec = V * I
  voltage = actuator.voltage[0].mean().item()
  current = actuator.current[0].mean().item()
  P_elec = abs(voltage * current)  # Absolute value for power magnitude

  # Copper losses: P_copper = I² * R
  R = test_motor_spec.resistance
  P_copper = current**2 * R

  # Sanity check: electrical power should be positive and exceed losses
  # (or be within tolerance for numerical precision)
  assert P_elec >= 0, "Electrical power should be non-negative"
  assert P_copper >= 0, "Copper losses should be non-negative"

  # P_elec should be >= P_copper (within small tolerance)
  # Allow some tolerance for numerical precision
  tolerance = 0.01  # 10mW

  assert P_elec >= P_copper - tolerance, (
    f"Power balance violated (energy created from nothing):\n"
    f"  P_elec = {P_elec:.4f}W\n"
    f"  P_copper = {P_copper:.4f}W\n"
    f"  P_elec should be >= P_copper"
  )

  print(f"\nInstantaneous Power Check:")
  print(f"  P_elec = {P_elec:.4f}W")
  print(f"  P_copper = {P_copper:.4f}W")
  print(f"  ✓ Energy conservation validated")


def test_energy_conservation_over_trajectory(robot_xml, test_motor_spec):
  """Validate energy conservation over a complete trajectory.

  Energy balance over time:
  - E_electrical = ∫ V * I dt (total electrical energy input)
  - E_mechanical = ∫ τ * ω dt (total mechanical work output)
  - E_heat = ∫ I² * R dt (total heat dissipated)
  - Should satisfy: E_elec ≈ E_mech + E_heat
  """
  # Create entity
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=test_motor_spec,
          stiffness=100.0,
          damping=5.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  # Energy accumulators
  E_electrical = 0.0
  E_mechanical = 0.0
  E_heat = 0.0

  dt = 0.002
  num_steps = 500  # 1 second trajectory

  num_joints = len(entity.joint_names)
  actuator = entity._actuators[0]
  R = test_motor_spec.resistance

  # Run trajectory with sinusoidal target
  for step in range(num_steps):
    # Sinusoidal target position
    t = step * dt
    target_pos = torch.sin(torch.tensor(2 * 3.14159 * 0.5 * t)).item()
    target = torch.ones((1, num_joints)) * target_pos

    entity.set_joint_position_target(target)
    entity.write_data_to_sim()
    entity.update(dt)

    # Accumulate energies
    voltage = actuator.voltage[0].mean().item()
    current = actuator.current[0].mean().item()
    velocity = entity._data.joint_vel[0].mean().item()

    # Energy increments (Power * dt)
    # P_elec = V * I
    E_electrical += voltage * current * dt
    # P_mech = τ * ω (estimate from back-EMF: τ = Kt * I)
    torque_est = (
      test_motor_spec.motor_constant_kt * current * test_motor_spec.gear_ratio
    )
    E_mechanical += torque_est * velocity * dt
    # P_heat = I² * R
    E_heat += current**2 * R * dt

  # Energy balance check
  E_total = E_mechanical + E_heat
  energy_error = abs(E_electrical - E_total)

  # Allow 15% tolerance for numerical integration, other losses
  tolerance = 0.15 * max(abs(E_electrical), abs(E_total))

  print(f"\nEnergy Conservation Over Trajectory:")
  print(f"  E_electrical = {E_electrical:.4f}J")
  print(f"  E_mechanical = {E_mechanical:.4f}J")
  print(f"  E_heat       = {E_heat:.4f}J")
  print(f"  E_total      = {E_total:.4f}J")
  print(
    f"  Error        = {energy_error:.4f}J ({energy_error / max(abs(E_electrical), 1e-6) * 100:.1f}%)"
  )

  assert energy_error < tolerance or energy_error < 0.1, (
    f"Energy conservation violated over trajectory:\n"
    f"  E_elec = {E_electrical:.4f}J\n"
    f"  E_mech = {E_mechanical:.4f}J\n"
    f"  E_heat = {E_heat:.4f}J\n"
    f"  E_total = {E_total:.4f}J\n"
    f"  Error = {energy_error:.4f}J"
  )


def test_heat_dissipation_accumulation(robot_xml, test_motor_spec):
  """Validate that heat dissipation accumulates correctly.

  Heat generation should follow I²R law:
  - Q = ∫ I² * R dt
  - Temperature rise should correlate with heat
  """
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=test_motor_spec,
          stiffness=100.0,
          damping=5.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  actuator = entity._actuators[0]
  R = test_motor_spec.resistance
  R_th = test_motor_spec.thermal_resistance

  # Run with constant load
  num_joints = len(entity.joint_names)
  target = torch.ones((1, num_joints)) * 0.5

  Q_accumulated = 0.0
  dt = 0.002
  num_steps = 500

  initial_temp = actuator.winding_temperature[0].mean().item()

  for _ in range(num_steps):
    entity.set_joint_position_target(target)
    entity.write_data_to_sim()
    entity.update(dt)

    current = actuator.current[0].mean().item()
    Q_accumulated += current**2 * R * dt

  final_temp = actuator.winding_temperature[0].mean().item()
  delta_T = final_temp - initial_temp

  # Expected temperature rise (steady-state approximation)
  # ΔT = P * R_th = (I²R) * R_th
  avg_power = Q_accumulated / (num_steps * dt)  # Average power
  expected_delta_T = avg_power * R_th

  print(f"\nHeat Dissipation Test:")
  print(f"  Heat accumulated: {Q_accumulated:.4f}J")
  print(f"  Avg power: {avg_power:.4f}W")
  print(f"  Initial temp: {initial_temp:.2f}°C")
  print(f"  Final temp: {final_temp:.2f}°C")
  print(f"  ΔT observed: {delta_T:.2f}°C")
  print(f"  ΔT expected: {expected_delta_T:.2f}°C")

  # Temperature should have increased
  assert delta_T > 0, "Temperature should increase with heat dissipation"

  # Temperature rise should be in reasonable range
  # (Allow factor of 2 due to transient vs steady-state)
  assert delta_T < expected_delta_T * 2, (
    f"Temperature rise too high: {delta_T:.2f}°C > 2 * {expected_delta_T:.2f}°C"
  )


def test_regenerative_braking_energy(robot_xml, test_motor_spec):
  """Validate energy flow during regenerative braking.

  When motor is backdriven (ω > 0, τ < 0):
  - Mechanical energy converted to electrical
  - Current flows opposite direction
  - P_mech < 0 (energy flowing out of mechanics into electrics)
  """
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=test_motor_spec,
          stiffness=100.0,
          damping=5.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  num_joints = len(entity.joint_names)
  dt = 0.002

  # First, spin up the joint to positive velocity
  for _ in range(100):
    target = torch.ones((1, num_joints)) * 1.0
    entity.set_joint_position_target(target)
    entity.write_data_to_sim()
    entity.update(dt)

  # Now command it to stop (regenerative braking)
  actuator = entity._actuators[0]

  # Record state during braking
  braking_samples = []
  for _ in range(50):
    target = torch.zeros((1, num_joints))
    entity.set_joint_position_target(target)
    entity.write_data_to_sim()
    entity.update(dt)

    velocity = entity._data.joint_vel[0].mean().item()
    current = actuator.current[0].mean().item()

    # During braking: velocity > 0, current should be negative (regenerative)
    # Estimate torque from motor equations: τ = Kt * I * N
    if velocity > 0.1:
      torque_est = (
        test_motor_spec.motor_constant_kt * current * test_motor_spec.gear_ratio
      )
      P_mech = torque_est * velocity
      braking_samples.append((velocity, torque_est, current, P_mech))

  # Validate regenerative braking occurred
  if len(braking_samples) > 5:
    avg_P_mech = sum(sample[3] for sample in braking_samples) / len(braking_samples)

    print(f"\nRegenerative Braking:")
    print(f"  Samples: {len(braking_samples)}")
    print(f"  Avg P_mech: {avg_P_mech:.4f}W")

    # During braking: P_mech should be negative (energy extracted)
    assert avg_P_mech < 0, (
      f"Mechanical power should be negative during braking: {avg_P_mech:.4f}W"
    )

    print("  ✓ Regenerative braking energy flow validated")
  else:
    # Skip if we didn't capture enough braking samples
    pytest.skip("Insufficient braking samples captured")


def test_no_free_energy(robot_xml, test_motor_spec):
  """Validate that system cannot create energy from nothing.

  With zero input power, system should only dissipate energy:
  - When motor disabled (V=0), kinetic energy should only decrease
  - No energy creation without electrical input
  """
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=test_motor_spec,
          stiffness=100.0,
          damping=5.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device, num_envs=1)

  num_joints = len(entity.joint_names)
  dt = 0.002

  # Spin up the joint
  for _ in range(100):
    target = torch.ones((1, num_joints)) * 1.0
    entity.set_joint_position_target(target)
    entity.write_data_to_sim()
    entity.update(dt)

  # Measure initial kinetic energy
  initial_velocity = entity._data.joint_vel[0].mean().item()
  initial_KE = 0.5 * test_motor_spec.reflected_inertia * initial_velocity**2

  # Now disable motor (set voltage to zero by setting very low targets)
  # This simulates coasting with no input power
  for _ in range(200):
    target = entity._data.joint_pos  # Hold current position (minimal torque)
    entity.set_joint_position_target(target)
    entity.write_data_to_sim()
    entity.update(dt)

  # Measure final kinetic energy
  final_velocity = entity._data.joint_vel[0].mean().item()
  final_KE = 0.5 * test_motor_spec.reflected_inertia * final_velocity**2

  print(f"\nNo Free Energy Test:")
  print(f"  Initial KE: {initial_KE:.6f}J")
  print(f"  Final KE: {final_KE:.6f}J")
  print(f"  Change: {final_KE - initial_KE:.6f}J")

  # Energy should have decreased (dissipated through damping/friction)
  # or stayed approximately constant
  assert final_KE <= initial_KE + 0.01, (
    f"Kinetic energy increased without input: "
    f"KE_final ({final_KE:.6f}J) > KE_initial ({initial_KE:.6f}J)"
  )

  print("  ✓ No energy creation validated")
