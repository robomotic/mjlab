"""Unit tests for electrical motor and battery metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs.mdp.metrics import (
  battery_current,
  battery_power,
  battery_soc,
  battery_temperature,
  battery_voltage,
  electrical_metrics_preset,
  motor_back_emf_avg,
  motor_current_avg,
  motor_power_total,
  motor_temperature_max,
  motor_voltage_avg,
)


@pytest.fixture
def mock_env():
  """Create a mock ManagerBasedRlEnv for testing."""
  env = MagicMock()
  env.num_envs = 4
  env.device = "cpu"
  return env


@pytest.fixture
def mock_scene_no_motors(mock_env):
  """Create a mock scene with no electrical motors or battery."""
  mock_env.scene = MagicMock()
  mock_env.scene.entities = {}
  mock_env.scene._battery_manager = None
  return mock_env


@pytest.fixture
def mock_scene_with_motors(mock_env):
  """Create a mock scene with electrical motors."""
  from mjlab.actuator import ElectricalMotorActuator

  # Create mock entity with electrical motor actuators
  entity = MagicMock()

  # Create mock actuator with electrical properties
  actuator1 = MagicMock(spec=ElectricalMotorActuator)
  actuator1.current = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  actuator1.voltage = torch.tensor(
    [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]
  )
  actuator1.power_dissipation = torch.tensor(
    [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]
  )
  actuator1.winding_temperature = torch.tensor(
    [[50.0, 55.0], [60.0, 65.0], [70.0, 75.0], [80.0, 85.0]]
  )
  actuator1.back_emf = torch.tensor([[5.0, 5.5], [6.0, 6.5], [7.0, 7.5], [8.0, 8.5]])

  actuator2 = MagicMock(spec=ElectricalMotorActuator)
  actuator2.current = torch.tensor([[10.0], [11.0], [12.0], [13.0]])
  actuator2.voltage = torch.tensor([[20.0], [21.0], [22.0], [23.0]])
  actuator2.power_dissipation = torch.tensor([[2.0], [2.1], [2.2], [2.3]])
  actuator2.winding_temperature = torch.tensor([[90.0], [95.0], [100.0], [105.0]])
  actuator2.back_emf = torch.tensor([[10.0], [10.5], [11.0], [11.5]])

  # Non-electrical actuator (should be ignored)
  actuator3 = MagicMock()
  actuator3.current = None

  entity._actuators = [actuator1, actuator2, actuator3]

  mock_env.scene.entities = {"robot": entity}
  mock_env.scene._battery_manager = None
  return mock_env


@pytest.fixture
def mock_scene_with_battery(mock_env):
  """Create a mock scene with battery manager."""
  mock_env.scene = MagicMock()
  mock_env.scene.entities = {}

  battery = MagicMock()
  battery.soc = torch.tensor([0.8, 0.75, 0.9, 0.85])
  battery.voltage = torch.tensor([48.0, 47.5, 48.5, 48.2])
  battery.current = torch.tensor([10.0, 12.0, 8.0, 11.0])
  battery.power_out = torch.tensor([480.0, 570.0, 388.0, 530.2])
  battery.temperature = torch.tensor([35.0, 36.0, 34.0, 35.5])

  mock_env.scene._battery_manager = battery
  return mock_env


# Motor Metrics Tests


def test_motor_current_avg_no_motors(mock_scene_no_motors):
  """Should return zeros when no electrical motors present."""
  result = motor_current_avg(mock_scene_no_motors)
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


def test_motor_current_avg_with_motors(mock_scene_with_motors):
  """Should return average current across all motors."""
  result = motor_current_avg(mock_scene_with_motors)
  assert result.shape == (4,)
  # Average of abs([1,2,10]), abs([3,4,11]), abs([5,6,12]), abs([7,8,13])
  expected = torch.tensor([4.333333, 6.0, 7.666667, 9.333333])
  assert torch.allclose(result, expected, atol=1e-5)


def test_motor_voltage_avg_with_motors(mock_scene_with_motors):
  """Should return average voltage across all motors."""
  result = motor_voltage_avg(mock_scene_with_motors)
  assert result.shape == (4,)
  # Average of abs([10,11,20]), abs([12,13,21]), abs([14,15,22]), abs([16,17,23])
  expected = torch.tensor([13.666667, 15.333333, 17.0, 18.666667])
  assert torch.allclose(result, expected, atol=1e-5)


def test_motor_power_total_with_motors(mock_scene_with_motors):
  """Should return total power dissipation across all motors."""
  result = motor_power_total(mock_scene_with_motors)
  assert result.shape == (4,)
  # Sum of [0.5,0.6,2.0], [0.7,0.8,2.1], [0.9,1.0,2.2], [1.1,1.2,2.3]
  expected = torch.tensor([3.1, 3.6, 4.1, 4.6])
  assert torch.allclose(result, expected, atol=1e-5)


def test_motor_temperature_max_with_motors(mock_scene_with_motors):
  """Should return maximum temperature across all motors."""
  result = motor_temperature_max(mock_scene_with_motors)
  assert result.shape == (4,)
  # Max of [50,55,90], [60,65,95], [70,75,100], [80,85,105]
  expected = torch.tensor([90.0, 95.0, 100.0, 105.0])
  assert torch.allclose(result, expected)


def test_motor_back_emf_avg_with_motors(mock_scene_with_motors):
  """Should return average back-EMF across all motors."""
  result = motor_back_emf_avg(mock_scene_with_motors)
  assert result.shape == (4,)
  # Average of abs([5.0,5.5,10.0]), abs([6.0,6.5,10.5]), abs([7.0,7.5,11.0]), abs([8.0,8.5,11.5])
  expected = torch.tensor([6.833333, 7.666667, 8.5, 9.333333])
  assert torch.allclose(result, expected, atol=1e-5)


# Battery Metrics Tests


def test_battery_soc_no_battery(mock_scene_no_motors):
  """Should return zeros when no battery present."""
  result = battery_soc(mock_scene_no_motors)
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


def test_battery_soc_with_battery(mock_scene_with_battery):
  """Should return battery SOC value."""
  result = battery_soc(mock_scene_with_battery)
  expected = torch.tensor([0.8, 0.75, 0.9, 0.85])
  assert torch.allclose(result, expected)


def test_battery_voltage_with_battery(mock_scene_with_battery):
  """Should return battery voltage value."""
  result = battery_voltage(mock_scene_with_battery)
  expected = torch.tensor([48.0, 47.5, 48.5, 48.2])
  assert torch.allclose(result, expected)


def test_battery_current_with_battery(mock_scene_with_battery):
  """Should return battery current value."""
  result = battery_current(mock_scene_with_battery)
  expected = torch.tensor([10.0, 12.0, 8.0, 11.0])
  assert torch.allclose(result, expected)


def test_battery_power_with_battery(mock_scene_with_battery):
  """Should return battery power value."""
  result = battery_power(mock_scene_with_battery)
  expected = torch.tensor([480.0, 570.0, 388.0, 530.2])
  assert torch.allclose(result, expected)


def test_battery_temperature_with_battery(mock_scene_with_battery):
  """Should return battery temperature value."""
  result = battery_temperature(mock_scene_with_battery)
  expected = torch.tensor([35.0, 36.0, 34.0, 35.5])
  assert torch.allclose(result, expected)


# Preset Tests


def test_electrical_metrics_preset_all():
  """Should return dict of all metrics."""
  from mjlab.managers import MetricsTermCfg

  result = electrical_metrics_preset()
  assert isinstance(result, dict)
  assert len(result) == 10

  # Verify motor metrics present
  assert "motor_current_avg" in result
  assert "motor_voltage_avg" in result
  assert "motor_power_total" in result
  assert "motor_temperature_max" in result
  assert "motor_back_emf_avg" in result

  # Verify battery metrics present
  assert "battery_soc" in result
  assert "battery_voltage" in result
  assert "battery_current" in result
  assert "battery_power" in result
  assert "battery_temperature" in result

  # Verify all values are MetricsTermCfg
  for value in result.values():
    assert isinstance(value, MetricsTermCfg)
    assert callable(value.func)


def test_electrical_metrics_preset_motor_only():
  """Should return only motor metrics when include_battery=False."""
  result = electrical_metrics_preset(include_motor=True, include_battery=False)
  assert len(result) == 5
  assert "motor_current_avg" in result
  assert "battery_soc" not in result


def test_electrical_metrics_preset_battery_only():
  """Should return only battery metrics when include_motor=False."""
  result = electrical_metrics_preset(include_motor=False, include_battery=True)
  assert len(result) == 5
  assert "battery_soc" in result
  assert "motor_current_avg" not in result


def test_electrical_metrics_preset_custom_entity():
  """Should accept custom entity_name parameter."""
  result = electrical_metrics_preset(entity_name="my_robot")
  assert "motor_current_avg" in result
  # Verify entity_name is passed as parameter
  assert result["motor_current_avg"].params == {"entity_name": "my_robot"}


def test_motor_metrics_with_no_entity(mock_env):
  """Should return zeros when entity doesn't exist."""
  mock_env.scene = MagicMock()
  mock_env.scene.entities = {}
  result = motor_current_avg(mock_env, entity_name="nonexistent")
  assert result.shape == (4,)
  assert torch.all(result == 0.0)


def test_metrics_return_correct_shape(mock_scene_with_motors, mock_scene_with_battery):
  """All metrics should return shape (B,)."""
  # Motor metrics
  assert motor_current_avg(mock_scene_with_motors).shape == (4,)
  assert motor_voltage_avg(mock_scene_with_motors).shape == (4,)
  assert motor_power_total(mock_scene_with_motors).shape == (4,)
  assert motor_temperature_max(mock_scene_with_motors).shape == (4,)
  assert motor_back_emf_avg(mock_scene_with_motors).shape == (4,)

  # Battery metrics
  assert battery_soc(mock_scene_with_battery).shape == (4,)
  assert battery_voltage(mock_scene_with_battery).shape == (4,)
  assert battery_current(mock_scene_with_battery).shape == (4,)
  assert battery_power(mock_scene_with_battery).shape == (4,)
  assert battery_temperature(mock_scene_with_battery).shape == (4,)
