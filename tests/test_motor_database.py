"""Tests for motor database module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mjlab.motor_database import (
  MotorSpecification,
  add_motor_database_path,
  get_default_search_paths,
  load_motor_spec,
)
from mjlab.motor_database.database import BUILTIN_MOTORS_PATH

# --- Motor Spec Tests ---


def test_motor_spec_creation():
  """Test creating MotorSpecification from all fields."""
  motor = MotorSpecification(
    motor_id="test_motor",
    manufacturer="TestCorp",
    model="TEST-1",
    gear_ratio=10.0,
    reflected_inertia=0.001,
    resistance=0.5,
    inductance=0.0001,
    motor_constant_kt=0.1,
    motor_constant_ke=0.1,
    stall_torque=15.0,
    continuous_torque=12.0,
  )

  assert motor.motor_id == "test_motor"
  assert motor.manufacturer == "TestCorp"
  assert motor.model == "TEST-1"
  assert motor.gear_ratio == 10.0
  assert motor.resistance == 0.5
  assert motor.stall_torque == 15.0


def test_motor_spec_optional_fields():
  """Test MotorSpecification with optional fields and defaults."""
  motor = MotorSpecification(
    motor_id="minimal_motor",
    manufacturer="MinCorp",
    model="MIN-1",
  )

  # Check defaults are applied
  assert motor.gear_ratio == 1.0
  assert motor.reflected_inertia == 0.0
  assert motor.resistance == 1.0
  assert motor.inductance == 0.001
  assert motor.stall_torque == 10.0
  assert motor.feedback_sensors == ["position", "velocity"]
  assert motor.protocol_params == {}
  assert motor.step_file is None
  assert motor.stl_file is None


def test_motor_spec_json_roundtrip():
  """Test JSON serialization and deserialization roundtrip."""
  original = MotorSpecification(
    motor_id="roundtrip_test",
    manufacturer="Test",
    model="RT-1",
    gear_ratio=5.0,
    resistance=0.8,
    feedback_sensors=["position", "velocity", "current"],
    protocol_params={"baudrate": 115200},
  )

  # Convert to dict (simulates JSON serialization)
  import dataclasses

  data = dataclasses.asdict(original)

  # Create from dict (simulates JSON deserialization)
  restored = MotorSpecification(**data)

  # Verify all fields match
  assert restored.motor_id == original.motor_id
  assert restored.manufacturer == original.manufacturer
  assert restored.gear_ratio == original.gear_ratio
  assert restored.resistance == original.resistance
  assert restored.feedback_sensors == original.feedback_sensors
  assert restored.protocol_params == original.protocol_params


def test_motor_spec_mutable_defaults():
  """Test that mutable defaults are properly initialized per instance."""
  motor1 = MotorSpecification(
    motor_id="motor1",
    manufacturer="Test",
    model="M1",
  )
  motor2 = MotorSpecification(
    motor_id="motor2",
    manufacturer="Test",
    model="M2",
  )

  # Modify motor1's list
  assert motor1.feedback_sensors is not None
  motor1.feedback_sensors.append("temperature")

  # Verify motor2 is unaffected
  assert motor2.feedback_sensors is not None
  assert len(motor2.feedback_sensors) == 2
  assert "temperature" not in motor2.feedback_sensors


# --- Database Loading Tests ---


def test_load_from_file(tmp_path):
  """Test loading motor from absolute file path."""
  # Create a test motor spec file
  motor_data = {
    "motor_id": "file_test_motor",
    "manufacturer": "FileTest",
    "model": "FT-1",
    "gear_ratio": 8.0,
    "resistance": 0.6,
  }

  motor_file = tmp_path / "test_motor.json"
  motor_file.write_text(json.dumps(motor_data))

  # Load motor
  motor = load_motor_spec(file=motor_file)

  assert motor.motor_id == "file_test_motor"
  assert motor.manufacturer == "FileTest"
  assert motor.gear_ratio == 8.0
  assert motor.resistance == 0.6


def test_load_from_builtin():
  """Test loading motor by ID from built-in database."""
  # Load the unitree_7520_14 motor (should exist in built-in database)
  motor = load_motor_spec("unitree_7520_14")

  assert motor.motor_id == "unitree_7520_14"
  assert motor.manufacturer == "Unitree"
  assert motor.model == "7520-14"
  assert motor.gear_ratio == 14.5
  assert motor.continuous_torque == 88.0


def test_load_from_builtin_alternate_motor():
  """Test loading another built-in motor."""
  motor = load_motor_spec("unitree_5020_9")

  assert motor.motor_id == "unitree_5020_9"
  assert motor.manufacturer == "Unitree"
  assert motor.continuous_torque == 25.0


def test_load_motor_not_found():
  """Test proper error when motor doesn't exist."""
  with pytest.raises(FileNotFoundError, match="not found in any search path"):
    load_motor_spec("nonexistent_motor_12345")


def test_load_from_url():
  """Test loading motor from URL with mocked network call."""
  motor_data = {
    "motor_id": "url_motor",
    "manufacturer": "URLTest",
    "model": "URL-1",
    "gear_ratio": 7.0,
  }

  # Mock urllib.request.urlopen
  mock_response = MagicMock()
  mock_response.read.return_value = json.dumps(motor_data).encode()
  mock_response.__enter__ = MagicMock(return_value=mock_response)
  mock_response.__exit__ = MagicMock(return_value=False)

  with patch("urllib.request.urlopen", return_value=mock_response):
    motor = load_motor_spec(url="https://example.com/motor.json")

  assert motor.motor_id == "url_motor"
  assert motor.manufacturer == "URLTest"
  assert motor.gear_ratio == 7.0


def test_search_path_priority(tmp_path):
  """Test search path priority (user > project > builtin)."""
  # Create a custom motor in a temporary directory
  custom_motor_data = {
    "motor_id": "unitree_7520_14",  # Same ID as builtin
    "manufacturer": "CustomManufacturer",  # Different manufacturer
    "model": "CUSTOM",
    "gear_ratio": 999.0,  # Easily distinguishable value
  }

  custom_dir = tmp_path / "custom_motors"
  custom_dir.mkdir()
  motor_file = custom_dir / "unitree_7520_14.json"
  motor_file.write_text(json.dumps(custom_motor_data))

  # Add custom path (should take precedence over built-in)
  add_motor_database_path(custom_dir)

  # Load motor - should get custom version, not built-in
  motor = load_motor_spec("unitree_7520_14")

  assert motor.manufacturer == "CustomManufacturer"
  assert motor.gear_ratio == 999.0

  # Clean up global state
  from mjlab.motor_database.database import _SEARCH_PATHS

  _SEARCH_PATHS.clear()


# --- Path Management Tests ---


def test_add_motor_database_path(tmp_path):
  """Test adding custom search path."""
  custom_dir = tmp_path / "my_motors"
  custom_dir.mkdir()

  # Should succeed for existing directory
  add_motor_database_path(custom_dir)

  from mjlab.motor_database.database import _SEARCH_PATHS

  assert custom_dir in _SEARCH_PATHS

  # Clean up
  _SEARCH_PATHS.clear()


def test_add_nonexistent_path(tmp_path):
  """Test error when adding non-existent path."""
  nonexistent = tmp_path / "does_not_exist"

  with pytest.raises(FileNotFoundError, match="does not exist"):
    add_motor_database_path(nonexistent)


def test_add_file_path(tmp_path):
  """Test error when adding a file instead of directory."""
  file_path = tmp_path / "not_a_dir.txt"
  file_path.write_text("test")

  with pytest.raises(NotADirectoryError, match="not a directory"):
    add_motor_database_path(file_path)


def test_get_default_search_paths():
  """Test that default search paths are returned correctly."""
  paths = get_default_search_paths()

  # Built-in path should always be present
  assert BUILTIN_MOTORS_PATH in paths

  # Paths should be Path objects
  assert all(isinstance(p, Path) for p in paths)


def test_environment_variable_path(tmp_path, monkeypatch):
  """Test MJLAB_MOTOR_PATH environment variable."""
  # Create test directories
  dir1 = tmp_path / "motors1"
  dir2 = tmp_path / "motors2"
  dir1.mkdir()
  dir2.mkdir()

  # Set environment variable (colon-separated)
  monkeypatch.setenv("MJLAB_MOTOR_PATH", f"{dir1}:{dir2}")

  paths = get_default_search_paths()

  # Both directories should be in search paths
  assert dir1 in paths
  assert dir2 in paths


def test_load_with_explicit_path(tmp_path):
  """Test loading motor with explicit path parameter."""
  # Create motor in custom location
  motor_data = {
    "motor_id": "explicit_path_motor",
    "manufacturer": "PathTest",
    "model": "PT-1",
  }

  custom_dir = tmp_path / "explicit"
  custom_dir.mkdir()
  motor_file = custom_dir / "explicit_path_motor.json"
  motor_file.write_text(json.dumps(motor_data))

  # Load with explicit path (should not search other paths)
  motor = load_motor_spec("explicit_path_motor", path=custom_dir)

  assert motor.motor_id == "explicit_path_motor"
  assert motor.manufacturer == "PathTest"


def test_multiple_source_error():
  """Test error when multiple sources are provided."""
  with pytest.raises(ValueError, match="Cannot combine"):
    load_motor_spec(motor_id="test", file="/some/path.json")

  with pytest.raises(ValueError, match="Cannot combine"):
    load_motor_spec(url="https://example.com", file="/some/path.json")


def test_no_source_error():
  """Test error when no source is provided."""
  with pytest.raises(ValueError, match="Must provide"):
    load_motor_spec()
