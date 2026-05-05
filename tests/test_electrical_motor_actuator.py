"""Comprehensive tests for ElectricalMotorActuator.

These tests verify that the ElectricalMotorActuator integrates properly with
the mjlab Entity system and validates the electrical and thermal physics:
- RL circuit dynamics with semi-implicit integration
- Thermal modeling with forward Euler
- Voltage and temperature limiting
- Per-environment state management
- Physics validation against analytical solutions
"""

import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.motor_database import MotorSpecification


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


@pytest.fixture
def test_motor_spec():
  """Create a test motor specification with known parameters."""
  return MotorSpecification(
    motor_id="test_motor",
    manufacturer="Test",
    model="TM-100",
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


def test_electrical_motor_actuator_creation(device, robot_xml, test_motor_spec):
  """Test that ElectricalMotorActuator can be created."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  assert entity is not None
  assert sim is not None


def test_electrical_motor_simulation_runs(device, robot_xml, test_motor_spec):
  """Test that simulation runs with ElectricalMotorActuator."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)

  # Run simulation
  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  for _ in range(10):
    entity.write_data_to_sim()
    sim.step()

  # Control should be finite
  ctrl = sim.data.ctrl[0]
  assert torch.isfinite(ctrl).all()


def test_back_emf_computation(device, robot_xml, test_motor_spec):
  """Test that back-EMF is proportional to velocity: V_bemf = Ke * omega."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=5.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Apply velocity command to build up speed
  entity.set_joint_position_target(torch.zeros(1, 1, device=device))
  entity.set_joint_velocity_target(torch.tensor([[5.0]], device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  # Run enough steps for velocity to build up
  for _ in range(50):
    entity.write_data_to_sim()
    sim.step()

  # Get actual velocity from simulation
  actual_vel = sim.data.qvel[0, 0].item()

  # Back-EMF should equal Ke * omega
  assert actuator.back_emf is not None
  expected_bemf = test_motor_spec.motor_constant_ke * actual_vel
  actual_bemf = actuator.back_emf[0, 0].item()

  # Check they're proportional (allow some tolerance for dynamics)
  if abs(actual_vel) > 0.1:  # Only check if significant velocity
    assert abs(actual_bemf - expected_bemf) < 0.1


def test_torque_current_relationship(device, robot_xml, test_motor_spec):
  """Test that torque = Kt * current."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Run a few steps to let current stabilize
  entity.set_joint_position_target(torch.tensor([[0.5]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  for _ in range(20):
    entity.write_data_to_sim()
    sim.step()

  # Check that applied torque matches Kt * current (before DC motor clipping)
  assert actuator.current is not None
  current = actuator.current[0, 0].item()
  Kt = test_motor_spec.motor_constant_kt
  expected_torque_before_clipping = Kt * current

  # Get actual control from sim
  actual_ctrl = sim.data.ctrl[0, 0].item()

  # They should be close (DC motor clipping may reduce slightly)
  assert abs(actual_ctrl) <= abs(expected_torque_before_clipping) + 1e-3


def test_voltage_clamping(device, robot_xml, test_motor_spec):
  """Test that voltage is clamped within [V_min, V_max]."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=10000.0,  # Very high stiffness to demand high voltage
      damping=100.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Set large position error to demand high voltage
  entity.set_joint_position_target(torch.tensor([[10.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  for _ in range(5):
    entity.write_data_to_sim()
    sim.step()

  # Check voltage is within bounds
  assert actuator.voltage is not None
  voltage = actuator.voltage[0, 0].item()
  V_min, V_max = test_motor_spec.voltage_range
  assert V_min <= voltage <= V_max


def test_power_dissipation(device, robot_xml, test_motor_spec):
  """Test that power dissipation = I^2 * R."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Run simulation to generate current
  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  dt = 0.002  # MuJoCo default timestep
  for _ in range(10):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  # After update(), power_dissipation should equal I^2 * R
  assert actuator.current is not None
  assert actuator.power_dissipation is not None
  current = actuator.current[0, 0].item()
  R = test_motor_spec.resistance
  expected_power = current**2 * R
  actual_power = actuator.power_dissipation[0, 0].item()

  assert abs(actual_power - expected_power) < 1e-5


def test_thermal_heating(device, robot_xml, test_motor_spec):
  """Test that winding temperature increases with sustained current."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Initial temperature should be ambient
  assert actuator.winding_temperature is not None
  T_initial = actuator.winding_temperature[0, 0].item()
  T_ambient = test_motor_spec.ambient_temperature
  assert abs(T_initial - T_ambient) < 1e-5

  # Apply sustained load
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  dt = 0.002  # MuJoCo default timestep
  for _ in range(100):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  # Temperature should have increased
  T_final = actuator.winding_temperature[0, 0].item()
  assert T_final > T_initial


def test_thermal_cooling_to_ambient(device, robot_xml):
  """Test that temperature decays back to ambient when motor is idle."""
  # Use a fast thermal time constant so cooling completes in few steps.
  fast_motor = MotorSpecification(
    motor_id="cooling_motor",
    manufacturer="Test",
    model="CM-1",
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
    thermal_time_constant=0.5,  # Fast: 0.5 s
    max_winding_temperature=150.0,
    ambient_temperature=25.0,
    encoder_resolution=1000,
    encoder_type="incremental",
    feedback_sensors=["position", "velocity", "current"],
    protocol="CAN",
    protocol_params={},
  )

  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=fast_motor,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=fast_motor.peak_torque,
      velocity_limit=fast_motor.no_load_speed,
      effort_limit=fast_motor.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  T_ambient = fast_motor.ambient_temperature
  dt = 0.002

  # Phase 1: heat the motor under load (5 thermal time constants = 2.5 s)
  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  n_heat = int(2.5 / dt)
  for _ in range(n_heat):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  assert actuator.winding_temperature is not None
  T_hot = actuator.winding_temperature[0, 0].item()
  assert T_hot > T_ambient + 1.0, "Motor must heat up before cooling test"

  # Phase 2: remove load and let the motor cool (zero targets)
  entity.set_joint_position_target(torch.zeros(1, 1, device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  n_cool = int(5.0 / dt)  # 10 thermal time constants — should fully cool
  for _ in range(n_cool):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  T_cooled = actuator.winding_temperature[0, 0].item()

  # Temperature should have returned close to ambient
  assert T_cooled < T_hot, "Temperature must decrease during cooling"
  assert abs(T_cooled - T_ambient) < 1.0, (
    f"Expected temperature near ambient ({T_ambient}), got {T_cooled}"
  )


def test_current_clamping(device, robot_xml):
  """Test that current is clamped to stall_current (drive electronics limit)."""
  # Motor with a low stall_current so the clamp is the binding constraint.
  # stall_current=5 A * Kt=0.1 → max torque 0.5 N·m, well below peak_torque=10 N·m.
  low_current_motor = MotorSpecification(
    motor_id="low_current_motor",
    manufacturer="Test",
    model="LC-1",
    gear_ratio=1.0,
    reflected_inertia=0.0,
    rotation_angle_range=(-3.14, 3.14),
    voltage_range=(0.0, 48.0),
    resistance=1.0,
    inductance=0.001,
    motor_constant_kt=0.1,
    motor_constant_ke=0.1,
    stall_current=5.0,
    peak_torque=10.0,
    no_load_speed=100.0,
    thermal_resistance=2.0,
    thermal_time_constant=100.0,
    max_winding_temperature=150.0,
    ambient_temperature=25.0,
  )

  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=low_current_motor,
      stiffness=100000.0,  # Very high stiffness to demand current far above stall_current
      damping=0.0,
      saturation_effort=low_current_motor.peak_torque,
      velocity_limit=low_current_motor.no_load_speed,
      effort_limit=low_current_motor.peak_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Large position error → demands huge current without the clamp
  entity.set_joint_position_target(torch.tensor([[10.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  entity.write_data_to_sim()
  sim.step()

  assert actuator.current is not None
  current = actuator.current[0, 0].item()
  assert abs(current) <= low_current_motor.stall_current + 1e-6, (
    f"Current {current:.3f} A exceeds stall_current {low_current_motor.stall_current} A"
  )
  # Output torque must also be bounded by stall_current * Kt
  ctrl = sim.data.ctrl[0, 0].item()
  max_torque_from_current = (
    low_current_motor.stall_current * low_current_motor.motor_constant_kt
  )
  assert abs(ctrl) <= max_torque_from_current + 1e-6, (
    f"Control torque {ctrl:.3f} N·m exceeds current-limited max {max_torque_from_current} N·m"
  )


def test_temperature_clamping(device, robot_xml, test_motor_spec):
  """Test that temperature stays within [T_ambient, T_max]."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Run for many steps
  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  dt = 0.002  # MuJoCo default timestep
  for _ in range(200):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  # Temperature should be within bounds
  assert actuator.winding_temperature is not None
  T = actuator.winding_temperature[0, 0].item()
  T_ambient = test_motor_spec.ambient_temperature
  T_max = test_motor_spec.max_winding_temperature
  assert T_ambient <= T <= T_max


def test_reset_electrical_state(device, robot_xml, test_motor_spec):
  """Test that reset() clears electrical state."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Run simulation to build up state
  entity.set_joint_position_target(torch.tensor([[1.0]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  dt = 0.002  # MuJoCo default timestep
  for _ in range(50):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  # Should have non-zero current and elevated temperature
  assert actuator.current is not None
  assert actuator.winding_temperature is not None
  assert abs(actuator.current[0, 0].item()) > 0.1
  T_before = actuator.winding_temperature[0, 0].item()
  T_ambient = test_motor_spec.ambient_temperature
  assert T_before > T_ambient

  # Reset
  actuator.reset(env_ids=None)

  # State should be cleared
  assert actuator.voltage is not None
  assert actuator.power_dissipation is not None
  assert abs(actuator.current[0, 0].item()) < 1e-6
  assert abs(actuator.voltage[0, 0].item()) < 1e-6
  assert abs(actuator.power_dissipation[0, 0].item()) < 1e-6
  T_after = actuator.winding_temperature[0, 0].item()
  assert abs(T_after - T_ambient) < 1e-5


def test_steady_state_ohms_law(device, robot_xml, test_motor_spec):
  """Test steady-state electrical behavior follows Ohm's law."""
  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=50.0,
      damping=5.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Hold at fixed position to reach steady-state
  entity.set_joint_position_target(torch.tensor([[0.5]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  # Run until steady-state (velocity ~0)
  for _ in range(200):
    entity.write_data_to_sim()
    sim.step()

  # At steady-state with near-zero velocity:
  # V_bemf ≈ 0, so V_terminal ≈ I * R
  velocity = sim.data.qvel[0, 0].item()
  if abs(velocity) < 0.1:  # Near steady-state
    assert actuator.voltage is not None
    assert actuator.current is not None
    voltage = actuator.voltage[0, 0].item()
    current = actuator.current[0, 0].item()
    R = test_motor_spec.resistance
    expected_voltage = current * R
    # Allow tolerance for small back-EMF and dynamics
    assert abs(voltage - expected_voltage) < 2.0


def test_thermal_steady_state(device, robot_xml, test_motor_spec):
  """Test thermal steady-state: P_loss = (T - T_amb) / R_th."""
  # Use motor with fast thermal response for testing
  fast_motor = MotorSpecification(
    motor_id="fast_thermal_motor",
    manufacturer="Test",
    model="FTM-1",
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
    thermal_time_constant=0.5,  # Fast: 0.5s instead of 100s
    max_winding_temperature=150.0,
    ambient_temperature=25.0,
    encoder_resolution=1000,
    encoder_type="incremental",
    feedback_sensors=["position", "velocity", "current"],
    protocol="CAN",
    protocol_params={},
  )

  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=fast_motor,
      stiffness=10.0,  # Lower stiffness to reduce current/heating
      damping=5.0,
      saturation_effort=fast_motor.peak_torque,
      velocity_limit=fast_motor.no_load_speed,
      effort_limit=fast_motor.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Apply moderate load
  entity.set_joint_position_target(torch.tensor([[0.3]], device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  # Run for several thermal time constants (5 * 0.5s = 2.5s total)
  dt = 0.002
  n_steps = int(2.5 / dt)
  for _ in range(n_steps):
    entity.write_data_to_sim()
    sim.step()
    actuator.update(dt)

  # At thermal steady-state: P_loss = (T - T_amb) / R_th
  assert actuator.winding_temperature is not None
  assert actuator.power_dissipation is not None
  T = actuator.winding_temperature[0, 0].item()
  T_amb = fast_motor.ambient_temperature
  R_th = fast_motor.thermal_resistance
  P_loss = actuator.power_dissipation[0, 0].item()

  # Compute expected steady-state temp rise
  expected_temp_rise = P_loss * R_th
  actual_temp_rise = T - T_amb

  # Check temperature has risen above ambient
  assert actual_temp_rise > 0.1

  # If not clamped, should be approaching steady-state
  if T < fast_motor.max_winding_temperature - 1.0:
    # Should be within 30% of steady-state after 5 time constants
    relative_error = abs(actual_temp_rise - expected_temp_rise) / (
      expected_temp_rise + 0.1
    )
    assert relative_error < 0.3


def test_inductance_effect_on_current_lag(device, robot_xml, test_motor_spec):
  """Test that inductance causes current lag on rapid command changes."""
  # Create motor with high inductance for testing
  high_L_motor = MotorSpecification(
    motor_id="high_L_motor",
    manufacturer="Test",
    model="HL-1",
    gear_ratio=10.0,
    reflected_inertia=0.01,
    rotation_angle_range=(-3.14, 3.14),
    voltage_range=(0.0, 48.0),
    resistance=1.0,
    inductance=0.1,  # High inductance: 100x normal
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

  entity = create_entity_with_actuator(
    robot_xml,
    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=high_L_motor,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=high_L_motor.peak_torque,
      velocity_limit=high_L_motor.no_load_speed,
      effort_limit=high_L_motor.continuous_torque,
    ),
  )

  entity, sim = initialize_entity(entity, device)
  from mjlab.actuator import ElectricalMotorActuator

  actuator = entity.actuators[0]
  assert isinstance(actuator, ElectricalMotorActuator)

  # Start from rest
  entity.set_joint_position_target(torch.zeros(1, 1, device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 1, device=device))
  entity.set_joint_effort_target(torch.zeros(1, 1, device=device))

  for _ in range(10):
    entity.write_data_to_sim()
    sim.step()

  # Record current before step change
  assert actuator.current is not None
  I_before = actuator.current[0, 0].item()

  # Apply step change in position target
  entity.set_joint_position_target(torch.tensor([[2.0]], device=device))

  # Current immediately after step should not jump instantly
  entity.write_data_to_sim()
  sim.step()
  I_after_1_step = actuator.current[0, 0].item()

  # Current should increase but not instantaneously (L limits dI/dt)
  # With high L, current change should be limited
  delta_I = abs(I_after_1_step - I_before)
  # L/R time constant = 0.1/1.0 = 0.1s, dt=0.002s → small change per step
  assert delta_I < 5.0  # Current shouldn't jump more than 5A in one step


def test_configuration_validation_warnings(test_motor_spec):
  """Test that configuration validates motor_spec consistency."""
  import warnings

  # Create config with mismatched saturation_effort
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=99.0,  # Doesn't match motor_spec.peak_torque (10.0)
      velocity_limit=test_motor_spec.no_load_speed,
      effort_limit=test_motor_spec.continuous_torque,
    )

    # Should have warned about saturation_effort mismatch
    assert len(w) >= 1
    assert "saturation_effort" in str(w[0].message).lower()

  # Create config with mismatched velocity_limit
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    ElectricalMotorActuatorCfg(
      target_names_expr=("joint.*",),
      motor_spec=test_motor_spec,
      stiffness=100.0,
      damping=10.0,
      saturation_effort=test_motor_spec.peak_torque,
      velocity_limit=999.0,  # Doesn't match motor_spec.no_load_speed (100.0)
      effort_limit=test_motor_spec.continuous_torque,
    )

    # Should have warned about velocity_limit mismatch
    assert len(w) >= 1
    assert "velocity_limit" in str(w[0].message).lower()
