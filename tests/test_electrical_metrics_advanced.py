"""Unit tests for advanced electrical metrics (per-joint and cumulative)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs.mdp.metrics import (
  CumulativeEnergyMetric,
  CumulativeMechanicalWorkMetric,
  motor_back_emf_joint,
  motor_current_joint,
  motor_power_joint,
  motor_temperature_joint,
  motor_voltage_joint,
)


@pytest.fixture
def mock_env():
  """Create a mock ManagerBasedRlEnv for testing."""
  env = MagicMock()
  env.num_envs = 4
  env.device = "cpu"
  env.physics_dt = 0.002
  env.cfg = MagicMock()
  env.cfg.decimation = 1
  return env


@pytest.fixture
def mock_scene_with_named_joints(mock_env):
  """Create a mock scene with electrical motors and named joints."""
  from mjlab.actuator import ElectricalMotorActuator

  entity = MagicMock()

  # Create mock actuator with electrical properties
  actuator = MagicMock(spec=ElectricalMotorActuator)
  actuator._target_names = ["joint1", "joint2"]
  actuator._target_indices = torch.tensor([0, 1])
  actuator.current = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  actuator.voltage = torch.tensor(
    [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]
  )
  actuator.power_dissipation = torch.tensor(
    [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]
  )
  actuator.winding_temperature = torch.tensor(
    [[50.0, 55.0], [60.0, 65.0], [70.0, 75.0], [80.0, 85.0]]
  )
  actuator.back_emf = torch.tensor([[5.0, 5.5], [6.0, 6.5], [7.0, 7.5], [8.0, 8.5]])

  # Mock motor spec
  actuator.motor_spec = MagicMock()
  actuator.motor_spec.motor_constant_kt = 0.1
  actuator.motor_spec.gear_ratio = 10.0

  entity._actuators = [actuator]
  entity._data = MagicMock()
  entity._data.joint_vel = torch.tensor(
    [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]]
  )

  mock_env.scene.entities = {"robot": entity}
  mock_env.scene._battery_manager = None
  return mock_env


@pytest.fixture
def mock_scene_with_battery(mock_env):
  """Create a mock scene with battery manager."""
  mock_env.scene = MagicMock()
  mock_env.scene.entities = {}

  battery = MagicMock()
  battery.soc = torch.tensor([1.0, 0.95, 0.9, 0.85])
  battery.voltage = torch.tensor([48.0, 47.5, 47.0, 46.5])
  battery.current = torch.tensor([10.0, 12.0, 14.0, 16.0])
  battery.power_out = torch.tensor([480.0, 570.0, 658.0, 744.0])
  battery.temperature = torch.tensor([25.0, 26.0, 27.0, 28.0])

  mock_env.scene._battery_manager = battery
  return mock_env


# Per-Joint Motor Metrics Tests


def test_motor_current_joint_valid(mock_scene_with_named_joints):
  """Should return current for a specific joint."""
  result = motor_current_joint(mock_scene_with_named_joints, joint_name="joint1")
  assert result.shape == (4,)
  # First joint (index 0): [1.0, 3.0, 5.0, 7.0]
  expected = torch.tensor([1.0, 3.0, 5.0, 7.0])
  assert torch.allclose(result, expected)


def test_motor_current_joint_second_joint(mock_scene_with_named_joints):
  """Should return current for the second joint."""
  result = motor_current_joint(mock_scene_with_named_joints, joint_name="joint2")
  assert result.shape == (4,)
  # Second joint (index 1): [2.0, 4.0, 6.0, 8.0]
  expected = torch.tensor([2.0, 4.0, 6.0, 8.0])
  assert torch.allclose(result, expected)


def test_motor_current_joint_not_found(mock_scene_with_named_joints):
  """Should return zeros for non-existent joint."""
  result = motor_current_joint(mock_scene_with_named_joints, joint_name="nonexistent")
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


def test_motor_current_joint_none(mock_scene_with_named_joints):
  """Should return zeros when joint_name is None."""
  result = motor_current_joint(mock_scene_with_named_joints, joint_name=None)
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


def test_motor_voltage_joint_valid(mock_scene_with_named_joints):
  """Should return voltage for a specific joint."""
  result = motor_voltage_joint(mock_scene_with_named_joints, joint_name="joint1")
  assert result.shape == (4,)
  expected = torch.tensor([10.0, 12.0, 14.0, 16.0])
  assert torch.allclose(result, expected)


def test_motor_power_joint_valid(mock_scene_with_named_joints):
  """Should return power for a specific joint."""
  result = motor_power_joint(mock_scene_with_named_joints, joint_name="joint2")
  assert result.shape == (4,)
  expected = torch.tensor([0.6, 0.8, 1.0, 1.2])
  assert torch.allclose(result, expected)


def test_motor_temperature_joint_valid(mock_scene_with_named_joints):
  """Should return temperature for a specific joint."""
  result = motor_temperature_joint(mock_scene_with_named_joints, joint_name="joint1")
  assert result.shape == (4,)
  expected = torch.tensor([50.0, 60.0, 70.0, 80.0])
  assert torch.allclose(result, expected)


def test_motor_back_emf_joint_valid(mock_scene_with_named_joints):
  """Should return back-EMF for a specific joint."""
  result = motor_back_emf_joint(mock_scene_with_named_joints, joint_name="joint2")
  assert result.shape == (4,)
  expected = torch.tensor([5.5, 6.5, 7.5, 8.5])
  assert torch.allclose(result, expected)


# Cumulative Energy Metric Tests


def test_cumulative_energy_initialization(mock_scene_with_battery):
  """Should initialize on first call and start accumulating."""
  metric = CumulativeEnergyMetric()
  result = metric(mock_scene_with_battery)
  assert result.shape == (4,)
  # First call initializes and accumulates energy from first step
  # Energy should be positive (battery is providing power)
  assert torch.all(result >= 0.0)


def test_cumulative_energy_accumulation(mock_scene_with_battery):
  """Should accumulate energy over multiple steps."""
  metric = CumulativeEnergyMetric()

  # First call initializes
  metric(mock_scene_with_battery)

  # Subsequent calls accumulate
  for _ in range(10):
    result = metric(mock_scene_with_battery)

  # Energy should be positive and increasing
  assert torch.all(result > 0.0)

  # Higher power environments should have more energy
  # power_out = [480.0, 570.0, 658.0, 744.0]
  assert result[3] > result[2] > result[1] > result[0]


def test_cumulative_energy_reset_all(mock_scene_with_battery):
  """Should reset all environments to zero."""
  metric = CumulativeEnergyMetric()

  # Accumulate some energy
  for _ in range(6):
    metric(mock_scene_with_battery)

  result_before = (
    metric._cumulative_energy.clone()
    if metric._cumulative_energy is not None
    else torch.zeros(4)
  )
  assert torch.all(result_before > 0.0)

  # Reset all
  final_values = metric.reset(env_ids=None)
  assert len(final_values) == 4

  # After reset, internal state should be zeros
  assert metric._cumulative_energy is not None
  assert torch.all(metric._cumulative_energy == 0.0)

  # Next call will accumulate from zero
  result_after = metric(mock_scene_with_battery)
  # Should be small (just one step of accumulation)
  assert torch.all(result_after < result_before)


def test_cumulative_energy_reset_specific(mock_scene_with_battery):
  """Should reset only specific environments."""
  metric = CumulativeEnergyMetric()

  # Accumulate energy
  for _ in range(6):
    metric(mock_scene_with_battery)

  result_before = (
    metric._cumulative_energy.clone()
    if metric._cumulative_energy is not None
    else torch.zeros(4)
  )

  # Reset only environments 1 and 2
  env_ids = torch.tensor([1, 2])
  final_values = metric.reset(env_ids=env_ids)
  assert len(final_values) == 2

  # Check internal state after reset
  assert metric._cumulative_energy is not None
  assert metric._cumulative_energy[0] == result_before[0]  # Not reset
  assert metric._cumulative_energy[1] == 0.0  # Reset
  assert metric._cumulative_energy[2] == 0.0  # Reset
  assert metric._cumulative_energy[3] == result_before[3]  # Not reset


def test_cumulative_energy_no_battery(mock_env):
  """Should return zeros when no battery present."""
  mock_env.scene = MagicMock()
  mock_env.scene._battery_manager = None

  metric = CumulativeEnergyMetric()
  result = metric(mock_env)
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


# Cumulative Mechanical Work Metric Tests


def test_cumulative_work_initialization(mock_scene_with_named_joints):
  """Should initialize on first call and start accumulating."""
  metric = CumulativeMechanicalWorkMetric()
  result = metric(mock_scene_with_named_joints)
  assert result.shape == (4,)
  # First call initializes and accumulates work from first step
  # Work can be positive or negative, but should be non-zero with our test data
  assert torch.any(result != 0.0)


def test_cumulative_work_accumulation(mock_scene_with_named_joints):
  """Should accumulate work over multiple steps."""
  metric = CumulativeMechanicalWorkMetric()

  # First call initializes
  metric(mock_scene_with_named_joints)

  # Subsequent calls accumulate
  for _ in range(10):
    result = metric(mock_scene_with_named_joints)

  # Work can be positive or negative depending on torque/velocity direction
  # But should be non-zero after multiple steps
  assert torch.any(result != 0.0)


def test_cumulative_work_reset_all(mock_scene_with_named_joints):
  """Should reset all environments to zero."""
  metric = CumulativeMechanicalWorkMetric()

  # Accumulate work
  for _ in range(6):
    metric(mock_scene_with_named_joints)

  result_before = (
    metric._cumulative_work.clone()
    if metric._cumulative_work is not None
    else torch.zeros(4)
  )

  # Reset all
  final_values = metric.reset(env_ids=None)
  assert len(final_values) == 4

  # After reset, internal state should be zeros
  assert metric._cumulative_work is not None
  assert torch.all(metric._cumulative_work == 0.0)

  # Next call will accumulate from zero
  result_after = metric(mock_scene_with_named_joints)
  # Should be small (just one step of accumulation)
  assert torch.all(torch.abs(result_after) < torch.abs(result_before))


def test_cumulative_work_no_entity(mock_env):
  """Should return zeros when entity doesn't exist."""
  mock_env.scene = MagicMock()
  mock_env.scene.entities = {}

  metric = CumulativeMechanicalWorkMetric()
  result = metric(mock_env)
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


def test_per_joint_metrics_return_correct_shape(mock_scene_with_named_joints):
  """All per-joint metrics should return shape (B,)."""
  assert motor_current_joint(
    mock_scene_with_named_joints, joint_name="joint1"
  ).shape == (4,)
  assert motor_voltage_joint(
    mock_scene_with_named_joints, joint_name="joint1"
  ).shape == (4,)
  assert motor_power_joint(mock_scene_with_named_joints, joint_name="joint1").shape == (
    4,
  )
  assert motor_temperature_joint(
    mock_scene_with_named_joints, joint_name="joint1"
  ).shape == (4,)
  assert motor_back_emf_joint(
    mock_scene_with_named_joints, joint_name="joint1"
  ).shape == (4,)
