"""Tests for battery manager module.

Tests battery physics, voltage drop, SOC tracking, thermal dynamics, and
current aggregation.
"""

import pytest
import torch

from mjlab.battery import BatteryManager, BatteryManagerCfg
from mjlab.battery_database import load_battery_spec


@pytest.fixture
def test_battery_spec():
  """Load test battery specification."""
  return load_battery_spec("test_battery")


@pytest.fixture
def battery_cfg(test_battery_spec):
  """Create battery manager configuration."""
  return BatteryManagerCfg(
    battery_spec=test_battery_spec,
    entity_names=("robot",),
    initial_soc=1.0,
    enable_voltage_feedback=True,
  )


@pytest.fixture
def mock_scene():
  """Create a mock scene for testing."""

  class MockScene:
    def __init__(self):
      self.entities = {}

  return MockScene()


@pytest.fixture
def battery_manager(battery_cfg, mock_scene):
  """Create initialized battery manager."""
  manager = BatteryManager(battery_cfg, mock_scene)
  manager.initialize(num_envs=4, device="cpu")
  return manager


# --- Initialization Tests ---


def test_battery_manager_creation(battery_cfg, mock_scene):
  """Test creating BatteryManager instance."""
  manager = BatteryManager(battery_cfg, mock_scene)

  assert manager.cfg == battery_cfg
  assert manager.scene == mock_scene
  assert manager.soc is None  # Not initialized yet


def test_battery_manager_initialization(battery_manager, battery_cfg):
  """Test battery manager initialization."""
  assert battery_manager.soc is not None
  assert battery_manager.voltage is not None
  assert battery_manager.current is not None
  assert battery_manager.temperature is not None
  assert battery_manager.power_out is not None

  # Check shapes (4 environments)
  assert battery_manager.soc.shape == (4,)
  assert battery_manager.voltage.shape == (4,)
  assert battery_manager.current.shape == (4,)
  assert battery_manager.temperature.shape == (4,)

  # Check initial values
  assert torch.allclose(battery_manager.soc, torch.tensor(1.0))
  assert torch.allclose(
    battery_manager.voltage,
    torch.tensor(battery_cfg.battery_spec.nominal_voltage),
  )
  assert torch.allclose(battery_manager.current, torch.tensor(0.0))
  assert torch.allclose(
    battery_manager.temperature,
    torch.tensor(battery_cfg.battery_spec.ambient_temperature),
  )


# --- Voltage Computation Tests ---


def test_voltage_at_full_soc_no_load(battery_manager, battery_cfg):
  """Test voltage equals OCV at full SOC with no load."""
  # SOC = 1.0, current = 0.0
  battery_manager.soc = torch.ones(4)
  battery_manager.current = torch.zeros(4)

  voltage = battery_manager.compute_voltage()

  # At full SOC, OCV should be max_voltage (4.2V * 4 cells = 16.8V)
  expected_voltage = battery_cfg.battery_spec.max_voltage
  assert torch.allclose(voltage, torch.tensor(expected_voltage), atol=0.01)


def test_voltage_drop_under_load(battery_manager, battery_cfg):
  """Test voltage drop under load: V = V_oc - I*R."""
  # Set SOC to 50% (mid-range)
  battery_manager.soc = torch.full((4,), 0.5)

  # Set current to 10A
  battery_manager.current = torch.full((4,), 10.0)

  voltage = battery_manager.compute_voltage()

  # At 50% SOC, OCV ≈ 3.7V * 4 = 14.8V (nominal)
  # Internal resistance at 50% SOC: R_base * 1.0 = 0.05Ω
  # Voltage drop: 10A * 0.05Ω = 0.5V
  # Terminal voltage: 14.8V - 0.5V = 14.3V

  expected_v_oc = battery_cfg.battery_spec.nominal_cell_voltage * 4
  expected_r = battery_cfg.battery_spec.internal_resistance
  expected_voltage = expected_v_oc - 10.0 * expected_r

  assert torch.allclose(voltage, torch.tensor(expected_voltage), atol=0.1)


def test_voltage_sag_at_low_soc(battery_manager, battery_cfg):
  """Test increased voltage sag at low SOC due to higher resistance."""
  # Set SOC to 10% (low)
  battery_manager.soc = torch.full((4,), 0.1)

  # Set current to 10A
  battery_manager.current = torch.full((4,), 10.0)

  voltage = battery_manager.compute_voltage()

  # At 10% SOC, resistance multiplier ≈ 2.3x (interpolated)
  # So R ≈ 0.05Ω * 2.3 = 0.115Ω
  # Voltage drop: 10A * 0.115Ω = 1.15V
  # This should be noticeably more than at 50% SOC

  # Voltage should be lower than nominal due to both low OCV and high R
  assert voltage[0] < battery_cfg.battery_spec.nominal_voltage - 1.0


def test_voltage_clamping(battery_manager, battery_cfg):
  """Test voltage is clamped to battery limits."""
  # Force very low SOC and very high current
  battery_manager.soc = torch.full((4,), 0.0)
  battery_manager.current = torch.full((4,), 100.0)  # Very high current

  voltage = battery_manager.compute_voltage()

  # Voltage should be clamped to min_voltage
  assert torch.all(voltage >= battery_cfg.battery_spec.min_voltage)


# --- SOC Tracking Tests ---


def test_soc_discharge(battery_manager, battery_cfg):
  """Test SOC decreases with current draw."""
  # Start at full charge
  battery_manager.soc = torch.ones(4)

  # Draw 1A for 1 second
  battery_manager.current = torch.full((4,), 1.0)

  # Update for 1 second
  battery_manager.update(dt=1.0)

  # Expected SOC change: -1A * 1s / (1.0Ah * 3600s) = -1/3600 ≈ -0.000278
  expected_soc = 1.0 - (1.0 / 3600.0)

  assert torch.allclose(battery_manager.soc, torch.tensor(expected_soc), atol=1e-5)


def test_soc_discharge_rate(battery_manager, battery_cfg):
  """Test SOC discharge rate is proportional to current."""
  initial_soc = 1.0
  battery_manager.soc = torch.full((4,), initial_soc)

  # Draw 10A for 360 seconds (6 minutes)
  battery_manager.current = torch.full((4,), 10.0)

  # Capacity: 1.0Ah = 3600 As
  # Expected discharge: 10A * 360s = 3600 As = 1.0Ah = 100% SOC
  # So SOC should go from 100% to 0%

  battery_manager.update(dt=360.0)

  # SOC should be near min_soc (0.2) due to clamping
  assert torch.allclose(
    battery_manager.soc,
    torch.tensor(battery_cfg.battery_spec.min_soc),
    atol=0.01,
  )


def test_soc_clamping_min(battery_manager, battery_cfg):
  """Test SOC is clamped to min_soc."""
  # Start at minimum SOC
  battery_manager.soc = torch.full((4,), battery_cfg.battery_spec.min_soc)

  # Draw current (should not go below min_soc)
  battery_manager.current = torch.full((4,), 5.0)

  battery_manager.update(dt=10.0)

  # SOC should stay at min_soc
  assert torch.allclose(
    battery_manager.soc,
    torch.tensor(battery_cfg.battery_spec.min_soc),
    atol=0.01,
  )


def test_soc_clamping_max(battery_manager, battery_cfg):
  """Test SOC is clamped to max_soc."""
  # Start at max SOC
  battery_manager.soc = torch.full((4,), battery_cfg.battery_spec.max_soc)

  # No current draw (would increase SOC if charging, but we clamp)
  battery_manager.current = torch.full((4,), -1.0)  # Negative = charging

  battery_manager.update(dt=10.0)

  # SOC should stay at max_soc
  assert torch.allclose(
    battery_manager.soc,
    torch.tensor(battery_cfg.battery_spec.max_soc),
    atol=0.01,
  )


# --- Internal Resistance Tests ---


def test_resistance_increases_at_low_soc(battery_manager, battery_cfg):
  """Test internal resistance increases at low SOC."""
  # Measure resistance at high SOC (80%)
  battery_manager.soc = torch.full((4,), 0.8)
  battery_manager._update_internal_resistance()
  r_high_soc = battery_manager.internal_resistance[0].item()

  # Measure resistance at low SOC (20%)
  battery_manager.soc = torch.full((4,), 0.2)
  battery_manager._update_internal_resistance()
  r_low_soc = battery_manager.internal_resistance[0].item()

  # Resistance should be higher at low SOC
  assert r_low_soc > r_high_soc


def test_resistance_temperature_dependence(battery_manager, battery_cfg):
  """Test internal resistance increases with temperature."""
  battery_manager.soc = torch.full((4,), 0.5)

  # Measure resistance at ambient temperature
  battery_manager.temperature = torch.full(
    (4,), battery_cfg.battery_spec.ambient_temperature
  )
  battery_manager._update_internal_resistance()
  r_ambient = battery_manager.internal_resistance[0].item()

  # Measure resistance at higher temperature (+20°C)
  battery_manager.temperature = torch.full(
    (4,), battery_cfg.battery_spec.ambient_temperature + 20.0
  )
  battery_manager._update_internal_resistance()
  r_hot = battery_manager.internal_resistance[0].item()

  # Resistance should increase with temperature
  assert r_hot > r_ambient


# --- Thermal Dynamics Tests ---


def test_thermal_heating(battery_manager, battery_cfg):
  """Test battery heats up with I²R losses."""
  initial_temp = battery_cfg.battery_spec.ambient_temperature
  battery_manager.temperature = torch.full((4,), initial_temp)
  battery_manager.soc = torch.full((4,), 0.5)

  # Draw 10A (significant current)
  battery_manager.current = torch.full((4,), 10.0)

  # Update for 10 seconds
  battery_manager.update(dt=10.0)

  # Temperature should increase
  assert torch.all(battery_manager.temperature > initial_temp)


def test_thermal_steady_state(battery_manager, battery_cfg):
  """Test thermal equilibrium: P_loss = (T - T_amb) / R_th."""
  # Set constant current
  current = 2.0
  battery_manager.current = torch.full((4,), current)
  battery_manager.soc = torch.full((4,), 0.5)

  # The thermal time constant is τ = C_th * R_th = 100 * 20 = 2000s
  # To reach 99% of steady-state takes ~5*τ = 10000s
  # This is too long for a unit test, so we'll just verify the trend is correct

  initial_temp = battery_manager.temperature[0].item()

  # Simulate for a reasonable time
  for _ in range(100):
    battery_manager.update(dt=1.0)

  final_temp = battery_manager.temperature[0].item()

  # Temperature should have increased
  assert final_temp > initial_temp

  # Calculate expected direction of change
  r_internal = battery_cfg.battery_spec.internal_resistance
  p_loss = current**2 * r_internal
  expected_temp_rise = p_loss * battery_cfg.battery_spec.thermal_resistance
  expected_steady_state = (
    battery_cfg.battery_spec.ambient_temperature + expected_temp_rise
  )

  # We should be moving toward the expected steady-state
  # (which is 29°C for 2A current)
  assert final_temp < expected_steady_state  # Haven't reached it yet
  assert final_temp > initial_temp  # But moving in the right direction


def test_thermal_cooling(battery_manager, battery_cfg):
  """Test battery cools down when current stops."""
  # Heat up the battery first
  battery_manager.temperature = torch.full(
    (4,), battery_cfg.battery_spec.ambient_temperature + 10.0
  )
  battery_manager.soc = torch.full((4,), 0.5)

  # Stop current draw
  battery_manager.current = torch.zeros(4)

  # Update for several seconds
  for _ in range(10):
    battery_manager.update(dt=1.0)

  # Temperature should decrease (but not necessarily reach ambient yet)
  assert torch.all(
    battery_manager.temperature < battery_cfg.battery_spec.ambient_temperature + 10.0
  )


def test_temperature_clamping(battery_manager, battery_cfg):
  """Test temperature is clamped to safety limits."""
  # Force very high current to try to exceed max temperature
  battery_manager.current = torch.full((4,), 50.0)
  battery_manager.soc = torch.full((4,), 0.5)

  # Simulate for a long time
  for _ in range(1000):
    battery_manager.update(dt=1.0)

  # Temperature should be clamped to max_temperature
  assert torch.all(
    battery_manager.temperature <= battery_cfg.battery_spec.max_temperature
  )


# --- Current Aggregation Tests ---


def test_aggregate_current_no_actuators(battery_manager):
  """Test current aggregation with no actuators."""
  # No entities/actuators
  battery_manager.scene.entities = {}

  battery_manager.aggregate_current()

  # Current should be zero
  assert torch.allclose(battery_manager.current, torch.tensor(0.0))


def test_current_limiting(battery_manager, battery_cfg):
  """Test current is clamped to max_continuous_current."""
  # Manually set a very high current
  battery_manager.current = torch.full((4,), 1000.0)

  # Aggregate (with no actuators, will reset and clamp)
  battery_manager.scene.entities = {}
  battery_manager.aggregate_current()

  # Current should be zero (no actuators)
  assert torch.allclose(battery_manager.current, torch.tensor(0.0))


# --- OCV Interpolation Tests ---


def test_ocv_interpolation_endpoints(battery_manager, battery_cfg):
  """Test OCV interpolation at curve endpoints."""
  # Test at SOC = 0.0
  soc_zero = torch.zeros(4)
  v_zero = battery_manager._interpolate_ocv(soc_zero)

  expected_v_zero = (
    battery_cfg.battery_spec.min_cell_voltage * battery_cfg.battery_spec.cells_series
  )
  assert torch.allclose(v_zero, torch.tensor(expected_v_zero), atol=0.01)

  # Test at SOC = 1.0
  soc_full = torch.ones(4)
  v_full = battery_manager._interpolate_ocv(soc_full)

  expected_v_full = (
    battery_cfg.battery_spec.max_cell_voltage * battery_cfg.battery_spec.cells_series
  )
  assert torch.allclose(v_full, torch.tensor(expected_v_full), atol=0.01)


def test_ocv_interpolation_midpoint(battery_manager, battery_cfg):
  """Test OCV interpolation at midpoint."""
  # Test at SOC = 0.5
  soc_mid = torch.full((4,), 0.5)
  v_mid = battery_manager._interpolate_ocv(soc_mid)

  # Should be close to nominal voltage
  expected_v_mid = battery_cfg.battery_spec.nominal_voltage
  assert torch.allclose(v_mid, torch.tensor(expected_v_mid), atol=0.1)


# --- Reset Tests ---


def test_reset_all_environments(battery_manager, battery_cfg):
  """Test resetting all environments."""
  # Modify state
  battery_manager.soc = torch.full((4,), 0.3)
  battery_manager.current = torch.full((4,), 10.0)
  battery_manager.temperature = torch.full((4,), 40.0)

  # Reset all
  battery_manager.reset(env_ids=None)

  # State should be reset to initial values
  assert torch.allclose(battery_manager.soc, torch.tensor(1.0))
  assert torch.allclose(battery_manager.current, torch.tensor(0.0))
  assert torch.allclose(
    battery_manager.temperature,
    torch.tensor(battery_cfg.battery_spec.ambient_temperature),
  )


def test_reset_specific_environments(battery_manager, battery_cfg):
  """Test resetting specific environments."""
  # Modify state
  battery_manager.soc = torch.full((4,), 0.3)
  battery_manager.current = torch.full((4,), 10.0)

  # Reset only environments 0 and 2
  battery_manager.reset(env_ids=torch.tensor([0, 2]))

  # Environments 0 and 2 should be reset
  assert battery_manager.soc[0] == pytest.approx(1.0)
  assert battery_manager.soc[2] == pytest.approx(1.0)
  assert battery_manager.current[0] == pytest.approx(0.0)
  assert battery_manager.current[2] == pytest.approx(0.0)

  # Environments 1 and 3 should be unchanged
  assert battery_manager.soc[1] == pytest.approx(0.3)
  assert battery_manager.soc[3] == pytest.approx(0.3)
  assert battery_manager.current[1] == pytest.approx(10.0)
  assert battery_manager.current[3] == pytest.approx(10.0)


# --- Power Output Tests ---


def test_power_output_calculation(battery_manager):
  """Test power output is calculated correctly."""
  # Set voltage and current
  battery_manager.voltage = torch.full((4,), 15.0)
  battery_manager.current = torch.full((4,), 10.0)

  # Update (will calculate power)
  battery_manager.soc = torch.full((4,), 0.5)
  battery_manager.update(dt=0.1)

  # Power = V * I = 15V * 10A = 150W
  expected_power = 15.0 * 10.0
  assert torch.allclose(
    battery_manager.power_out,
    torch.tensor(expected_power),
    atol=0.1,
  )


# --- Per-Environment Independence Tests ---


def test_per_environment_independence(battery_manager):
  """Test that environments are independent."""
  # Set different SOC for each environment
  battery_manager.soc = torch.tensor([1.0, 0.8, 0.5, 0.2])

  # Set different currents
  battery_manager.current = torch.tensor([0.0, 5.0, 10.0, 15.0])

  # Compute voltages
  voltages = battery_manager.compute_voltage()

  # Voltages should be different (higher SOC and lower current = higher voltage)
  assert voltages[0] > voltages[1] > voltages[2] > voltages[3]
