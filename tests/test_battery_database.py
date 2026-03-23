"""Tests for battery database module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mjlab.battery_database import (
  BatterySpecification,
  add_battery_database_path,
  get_default_search_paths,
  load_battery_spec,
)
from mjlab.battery_database.database import BUILTIN_BATTERIES_PATH

# --- Battery Spec Tests ---


def test_battery_spec_creation():
  """Test creating BatterySpecification from all fields."""
  battery = BatterySpecification(
    battery_id="test_battery",
    manufacturer="TestCorp",
    model="TEST-6S2P",
    chemistry="LiPo",
    cells_series=6,
    cells_parallel=2,
    nominal_cell_voltage=3.7,
    max_cell_voltage=4.2,
    min_cell_voltage=3.0,
    min_operating_voltage=3.3,
    capacity_ah=5.0,
    internal_resistance=0.01,
    internal_resistance_temp_coeff=0.0001,
    max_continuous_current=100.0,
    max_burst_current=200.0,
    burst_duration=10.0,
    c_rating_continuous=20.0,
    thermal_capacity=500.0,
    thermal_resistance=10.0,
    max_temperature=60.0,
    min_temperature=-10.0,
    ambient_temperature=25.0,
  )

  assert battery.battery_id == "test_battery"
  assert battery.manufacturer == "TestCorp"
  assert battery.chemistry == "LiPo"
  assert battery.cells_series == 6
  assert battery.cells_parallel == 2
  assert battery.capacity_ah == 5.0


def test_battery_spec_computed_properties():
  """Test that battery pack voltages are computed correctly."""
  battery = BatterySpecification(
    battery_id="computed_test",
    manufacturer="Test",
    model="4S",
    chemistry="LiPo",
    cells_series=4,
    cells_parallel=1,
    nominal_cell_voltage=3.7,
    max_cell_voltage=4.2,
    min_cell_voltage=3.0,
    min_operating_voltage=3.3,
    capacity_ah=1.0,
    internal_resistance=0.05,
    internal_resistance_temp_coeff=0.0001,
    max_continuous_current=20.0,
    max_burst_current=30.0,
    burst_duration=5.0,
    c_rating_continuous=20.0,
    thermal_capacity=100.0,
    thermal_resistance=20.0,
    max_temperature=60.0,
    min_temperature=-10.0,
    ambient_temperature=25.0,
  )

  # Pack voltages should be cell voltage * cells_series
  assert battery.nominal_voltage == 3.7 * 4
  assert battery.max_voltage == 4.2 * 4
  assert battery.min_voltage == 3.0 * 4

  # Energy should be computed
  assert battery.energy_wh == pytest.approx(1.0 * 3.7 * 4, rel=1e-6)


def test_battery_spec_default_curves():
  """Test that default OCV and resistance curves are created."""
  battery = BatterySpecification(
    battery_id="default_curves",
    manufacturer="Test",
    model="Simple",
    chemistry="LiPo",
    cells_series=4,
    cells_parallel=1,
    nominal_cell_voltage=3.7,
    max_cell_voltage=4.2,
    min_cell_voltage=3.0,
    min_operating_voltage=3.3,
    capacity_ah=1.0,
    internal_resistance=0.05,
    internal_resistance_temp_coeff=0.0001,
    max_continuous_current=20.0,
    max_burst_current=30.0,
    burst_duration=5.0,
    c_rating_continuous=20.0,
    thermal_capacity=100.0,
    thermal_resistance=20.0,
    max_temperature=60.0,
    min_temperature=-10.0,
    ambient_temperature=25.0,
  )

  # OCV curve should be created
  assert battery.ocv_curve is not None
  assert len(battery.ocv_curve) == 3
  assert battery.ocv_curve[0] == (0.0, 3.0)
  assert battery.ocv_curve[1] == (0.5, 3.7)
  assert battery.ocv_curve[2] == (1.0, 4.2)

  # Resistance-SOC curve should be created
  assert battery.internal_resistance_soc_curve is not None
  assert len(battery.internal_resistance_soc_curve) == 4


def test_battery_spec_json_roundtrip():
  """Test JSON serialization and deserialization roundtrip."""
  original = BatterySpecification(
    battery_id="roundtrip_test",
    manufacturer="Test",
    model="RT-6S",
    chemistry="LiPo",
    cells_series=6,
    cells_parallel=1,
    nominal_cell_voltage=3.7,
    max_cell_voltage=4.2,
    min_cell_voltage=3.0,
    min_operating_voltage=3.3,
    capacity_ah=2.5,
    internal_resistance=0.02,
    internal_resistance_temp_coeff=0.0001,
    max_continuous_current=50.0,
    max_burst_current=100.0,
    burst_duration=10.0,
    c_rating_continuous=20.0,
    thermal_capacity=300.0,
    thermal_resistance=15.0,
    max_temperature=60.0,
    min_temperature=-10.0,
    ambient_temperature=25.0,
    ocv_curve=[(0.0, 3.0), (0.5, 3.7), (1.0, 4.2)],
  )

  # Convert to dict (simulates JSON serialization)
  import dataclasses

  data = dataclasses.asdict(original)

  # Create from dict (simulates JSON deserialization)
  restored = BatterySpecification(**data)

  # Verify all fields match
  assert restored.battery_id == original.battery_id
  assert restored.manufacturer == original.manufacturer
  assert restored.chemistry == original.chemistry
  assert restored.cells_series == original.cells_series
  assert restored.capacity_ah == original.capacity_ah
  assert restored.ocv_curve == original.ocv_curve


def test_battery_spec_mutable_defaults():
  """Test that mutable defaults are properly initialized per instance."""
  battery1 = BatterySpecification(
    battery_id="battery1",
    manufacturer="Test",
    model="B1",
    chemistry="LiPo",
    cells_series=4,
    cells_parallel=1,
    nominal_cell_voltage=3.7,
    max_cell_voltage=4.2,
    min_cell_voltage=3.0,
    min_operating_voltage=3.3,
    capacity_ah=1.0,
    internal_resistance=0.05,
    internal_resistance_temp_coeff=0.0001,
    max_continuous_current=20.0,
    max_burst_current=30.0,
    burst_duration=5.0,
    c_rating_continuous=20.0,
    thermal_capacity=100.0,
    thermal_resistance=20.0,
    max_temperature=60.0,
    min_temperature=-10.0,
    ambient_temperature=25.0,
  )

  battery2 = BatterySpecification(
    battery_id="battery2",
    manufacturer="Test",
    model="B2",
    chemistry="LiFePO4",
    cells_series=4,
    cells_parallel=1,
    nominal_cell_voltage=3.2,
    max_cell_voltage=3.65,
    min_cell_voltage=2.5,
    min_operating_voltage=2.8,
    capacity_ah=1.0,
    internal_resistance=0.05,
    internal_resistance_temp_coeff=0.0001,
    max_continuous_current=20.0,
    max_burst_current=30.0,
    burst_duration=5.0,
    c_rating_continuous=20.0,
    thermal_capacity=100.0,
    thermal_resistance=20.0,
    max_temperature=60.0,
    min_temperature=-10.0,
    ambient_temperature=25.0,
  )

  # Modify battery1's curve
  assert battery1.ocv_curve is not None
  battery1.ocv_curve.append((0.9, 4.0))

  # Verify battery2 is unaffected
  assert battery2.ocv_curve is not None
  assert len(battery2.ocv_curve) == 3
  assert (0.9, 4.0) not in battery2.ocv_curve


# --- Database Loading Tests ---


def test_load_from_file(tmp_path):
  """Test loading battery from absolute file path."""
  battery_data = {
    "battery_id": "file_test_battery",
    "manufacturer": "FileTest",
    "model": "FT-4S",
    "chemistry": "LiPo",
    "cells_series": 4,
    "cells_parallel": 1,
    "nominal_cell_voltage": 3.7,
    "max_cell_voltage": 4.2,
    "min_cell_voltage": 3.0,
    "min_operating_voltage": 3.3,
    "capacity_ah": 2.0,
    "internal_resistance": 0.03,
    "internal_resistance_temp_coeff": 0.0001,
    "max_continuous_current": 40.0,
    "max_burst_current": 80.0,
    "burst_duration": 10.0,
    "c_rating_continuous": 20.0,
    "thermal_capacity": 200.0,
    "thermal_resistance": 15.0,
    "max_temperature": 60.0,
    "min_temperature": -10.0,
    "ambient_temperature": 25.0,
  }

  battery_file = tmp_path / "test_battery.json"
  battery_file.write_text(json.dumps(battery_data))

  battery = load_battery_spec(file=battery_file)

  assert battery.battery_id == "file_test_battery"
  assert battery.manufacturer == "FileTest"
  assert battery.capacity_ah == 2.0


def test_load_from_builtin():
  """Test loading battery by ID from built-in database."""
  battery = load_battery_spec("turnigy_6s2p_5000mah")

  assert battery.battery_id == "turnigy_6s2p_5000mah"
  assert battery.manufacturer == "Turnigy"
  assert battery.chemistry == "LiPo"
  assert battery.cells_series == 6
  assert battery.cells_parallel == 2
  assert battery.capacity_ah == 5.0


def test_load_from_builtin_lifepo4():
  """Test loading LiFePO4 battery."""
  battery = load_battery_spec("lifepo4_12s_10ah")

  assert battery.battery_id == "lifepo4_12s_10ah"
  assert battery.chemistry == "LiFePO4"
  assert battery.cells_series == 12
  assert battery.capacity_ah == 10.0
  assert battery.nominal_cell_voltage == 3.2


def test_load_battery_not_found():
  """Test proper error when battery doesn't exist."""
  with pytest.raises(FileNotFoundError, match="not found in search paths"):
    load_battery_spec("nonexistent_battery_12345")


def test_load_from_url():
  """Test loading battery from URL with mocked network call."""
  battery_data = {
    "battery_id": "url_battery",
    "manufacturer": "URLTest",
    "model": "URL-4S",
    "chemistry": "LiPo",
    "cells_series": 4,
    "cells_parallel": 1,
    "nominal_cell_voltage": 3.7,
    "max_cell_voltage": 4.2,
    "min_cell_voltage": 3.0,
    "min_operating_voltage": 3.3,
    "capacity_ah": 1.5,
    "internal_resistance": 0.04,
    "internal_resistance_temp_coeff": 0.0001,
    "max_continuous_current": 30.0,
    "max_burst_current": 60.0,
    "burst_duration": 10.0,
    "c_rating_continuous": 20.0,
    "thermal_capacity": 150.0,
    "thermal_resistance": 18.0,
    "max_temperature": 60.0,
    "min_temperature": -10.0,
    "ambient_temperature": 25.0,
  }

  mock_response = MagicMock()
  mock_response.read.return_value = json.dumps(battery_data).encode()
  mock_response.__enter__ = MagicMock(return_value=mock_response)
  mock_response.__exit__ = MagicMock(return_value=False)

  with patch("urllib.request.urlopen", return_value=mock_response):
    battery = load_battery_spec(url="https://example.com/battery.json")

  assert battery.battery_id == "url_battery"
  assert battery.manufacturer == "URLTest"
  assert battery.capacity_ah == 1.5


def test_search_path_priority(tmp_path):
  """Test search path priority (custom > builtin)."""
  custom_battery_data = {
    "battery_id": "turnigy_6s2p_5000mah",
    "manufacturer": "CustomManufacturer",
    "model": "CUSTOM",
    "chemistry": "LiPo",
    "cells_series": 6,
    "cells_parallel": 2,
    "nominal_cell_voltage": 3.7,
    "max_cell_voltage": 4.2,
    "min_cell_voltage": 3.0,
    "min_operating_voltage": 3.3,
    "capacity_ah": 999.0,
    "internal_resistance": 0.01,
    "internal_resistance_temp_coeff": 0.0001,
    "max_continuous_current": 100.0,
    "max_burst_current": 200.0,
    "burst_duration": 10.0,
    "c_rating_continuous": 20.0,
    "thermal_capacity": 500.0,
    "thermal_resistance": 10.0,
    "max_temperature": 60.0,
    "min_temperature": -10.0,
    "ambient_temperature": 25.0,
  }

  custom_dir = tmp_path / "custom_batteries"
  custom_dir.mkdir()
  battery_file = custom_dir / "turnigy_6s2p_5000mah.json"
  battery_file.write_text(json.dumps(custom_battery_data))

  add_battery_database_path(custom_dir)

  battery = load_battery_spec("turnigy_6s2p_5000mah")

  assert battery.manufacturer == "CustomManufacturer"
  assert battery.capacity_ah == 999.0

  # Clean up global state
  from mjlab.battery_database.database import _SEARCH_PATHS

  _SEARCH_PATHS.clear()


# --- Path Management Tests ---


def test_add_battery_database_path(tmp_path):
  """Test adding custom search path."""
  custom_dir = tmp_path / "my_batteries"
  custom_dir.mkdir()

  add_battery_database_path(custom_dir)

  from mjlab.battery_database.database import _SEARCH_PATHS

  assert custom_dir in _SEARCH_PATHS

  _SEARCH_PATHS.clear()


def test_add_nonexistent_path(tmp_path):
  """Test error when adding non-existent path."""
  nonexistent = tmp_path / "does_not_exist"

  with pytest.raises(FileNotFoundError, match="does not exist"):
    add_battery_database_path(nonexistent)


def test_add_file_path(tmp_path):
  """Test error when adding a file instead of directory."""
  file_path = tmp_path / "not_a_dir.txt"
  file_path.write_text("test")

  with pytest.raises(NotADirectoryError, match="not a directory"):
    add_battery_database_path(file_path)


def test_get_default_search_paths():
  """Test that default search paths are returned correctly."""
  paths = get_default_search_paths()

  assert BUILTIN_BATTERIES_PATH in paths
  assert all(isinstance(p, Path) for p in paths)


def test_environment_variable_path(tmp_path, monkeypatch):
  """Test MJLAB_BATTERY_PATH environment variable."""
  dir1 = tmp_path / "batteries1"
  dir2 = tmp_path / "batteries2"
  dir1.mkdir()
  dir2.mkdir()

  monkeypatch.setenv("MJLAB_BATTERY_PATH", f"{dir1}:{dir2}")

  paths = get_default_search_paths()

  assert dir1 in paths
  assert dir2 in paths


def test_load_with_explicit_path(tmp_path):
  """Test loading battery with explicit path parameter."""
  battery_data = {
    "battery_id": "explicit_path_battery",
    "manufacturer": "PathTest",
    "model": "PT-4S",
    "chemistry": "LiPo",
    "cells_series": 4,
    "cells_parallel": 1,
    "nominal_cell_voltage": 3.7,
    "max_cell_voltage": 4.2,
    "min_cell_voltage": 3.0,
    "min_operating_voltage": 3.3,
    "capacity_ah": 1.0,
    "internal_resistance": 0.05,
    "internal_resistance_temp_coeff": 0.0001,
    "max_continuous_current": 20.0,
    "max_burst_current": 30.0,
    "burst_duration": 5.0,
    "c_rating_continuous": 20.0,
    "thermal_capacity": 100.0,
    "thermal_resistance": 20.0,
    "max_temperature": 60.0,
    "min_temperature": -10.0,
    "ambient_temperature": 25.0,
  }

  custom_dir = tmp_path / "explicit"
  custom_dir.mkdir()
  battery_file = custom_dir / "explicit_path_battery.json"
  battery_file.write_text(json.dumps(battery_data))

  battery = load_battery_spec("explicit_path_battery", path=custom_dir)

  assert battery.battery_id == "explicit_path_battery"
  assert battery.manufacturer == "PathTest"


def test_multiple_source_error():
  """Test error when multiple sources are provided."""
  with pytest.raises(ValueError, match="Can only provide one"):
    load_battery_spec(battery_id="test", file="/some/path.json")

  with pytest.raises(ValueError, match="Can only provide one"):
    load_battery_spec(url="https://example.com", file="/some/path.json")


def test_no_source_error():
  """Test error when no source is provided."""
  with pytest.raises(ValueError, match="Must provide one"):
    load_battery_spec()
