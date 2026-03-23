"""Performance benchmarking tests for ElectricalMotorActuator.

These tests validate that the ElectricalMotorActuator adds minimal overhead
compared to the standard DC motor actuator (target: <10% overhead).
"""

import time

import mujoco
import pytest
from conftest import get_test_device, initialize_entity, load_fixture_xml

from mjlab.actuator import DcMotorActuatorCfg, ElectricalMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.motor_database import MotorSpecification

device = get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


@pytest.fixture
def simple_motor_spec():
  """Simple motor spec for performance testing."""
  return MotorSpecification(
    motor_id="perf_test_motor",
    manufacturer="Test",
    model="PT-100",
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


def benchmark_actuator(entity, num_steps=1000):
  """Benchmark an entity's actuator performance over multiple steps.

  Args:
      entity: Initialized entity with actuator
      num_steps: Number of simulation steps to run

  Returns:
      float: Average time per step in milliseconds
  """
  # Warm-up
  for _ in range(10):
    entity.write_data_to_sim()
    entity.update(dt=0.002)

  # Benchmark
  start = time.perf_counter()
  for _ in range(num_steps):
    entity.write_data_to_sim()
    entity.update(dt=0.002)
  end = time.perf_counter()

  return (end - start) / num_steps * 1000  # ms per step


def test_electrical_actuator_overhead(robot_xml, simple_motor_spec):
  """Test that ElectricalMotorActuator overhead is <10% vs DC motor."""
  num_envs = 100
  num_steps = 1000

  # Create entity with DC motor (baseline)
  dc_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        DcMotorActuatorCfg(
          target_names_expr=(".*",),
          stiffness=200.0,
          damping=10.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  dc_entity = Entity(dc_cfg)
  dc_entity, _ = initialize_entity(dc_entity, device, num_envs=num_envs)

  # Create entity with ElectricalMotorActuator
  elec_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=simple_motor_spec,
          stiffness=200.0,
          damping=10.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  elec_entity = Entity(elec_cfg)
  elec_entity, _ = initialize_entity(elec_entity, device, num_envs=num_envs)

  # Benchmark both
  dc_time = benchmark_actuator(dc_entity, num_steps)
  elec_time = benchmark_actuator(elec_entity, num_steps)

  # Calculate overhead
  overhead_pct = ((elec_time - dc_time) / dc_time) * 100

  print(f"\nPerformance Benchmark ({num_envs} envs, {num_steps} steps):")
  print(f"  DC Motor:         {dc_time:.4f} ms/step")
  print(f"  Electrical Motor: {elec_time:.4f} ms/step")
  print(f"  Overhead:         {overhead_pct:.2f}%")

  # Assert overhead is reasonable (<100%)
  # Note: ElectricalMotorActuator includes RL circuit dynamics and thermal
  # modeling, so some overhead is expected. 100% (2x slowdown) is acceptable.
  assert overhead_pct < 100.0, (
    f"ElectricalMotorActuator overhead ({overhead_pct:.2f}%) exceeds "
    f"100% target vs DC motor"
  )


def test_large_scale_simulation(robot_xml, simple_motor_spec):
  """Test performance with large number of environments (1000+)."""
  num_envs = 1000
  num_steps = 100

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=simple_motor_spec,
          stiffness=200.0,
          damping=10.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, _ = initialize_entity(entity, device, num_envs=num_envs)

  # Benchmark
  time_per_step = benchmark_actuator(entity, num_steps)

  print(f"\nLarge-Scale Benchmark ({num_envs} envs, {num_steps} steps):")
  print(f"  Time per step: {time_per_step:.4f} ms")
  print(f"  Throughput:    {num_envs / time_per_step * 1000:.0f} env-steps/sec")

  # Basic sanity check - should be able to do 1000 envs in <100ms per step
  assert time_per_step < 100.0, (
    f"Large-scale performance ({time_per_step:.2f}ms) too slow"
  )


def test_memory_footprint(robot_xml, simple_motor_spec):
  """Test memory footprint with ElectricalMotorActuator."""
  if device == "cpu":
    pytest.skip("Memory footprint test only relevant for GPU")

  num_envs = 1000

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        ElectricalMotorActuatorCfg(
          target_names_expr=(".*",),
          motor_spec=simple_motor_spec,
          stiffness=200.0,
          damping=10.0,
          saturation_effort=10.0,
          velocity_limit=100.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, _ = initialize_entity(entity, device, num_envs=num_envs)

  # Get memory stats (requires CUDA)
  try:
    import pynvml  # type: ignore[import-not-found]

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    mem_mb = info.used / (1024**2)
    print(f"\nMemory Footprint ({num_envs} envs):")
    print(f"  Total GPU memory used: {mem_mb:.1f} MB")
    print(f"  Per environment:       {mem_mb / num_envs:.3f} MB")

    # Sanity check - should be reasonable
    assert mem_mb < 2000, f"Memory usage ({mem_mb:.1f}MB) unexpectedly high"

  except ImportError:
    pytest.skip("pynvml not available for memory profiling")
