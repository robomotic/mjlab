"""Tests for automatic discovery of motor and battery specs from XML."""

import mujoco
import pytest

from mjlab.entity import Entity, EntityCfg
from mjlab.scene import Scene, SceneCfg

from conftest import get_test_device, initialize_entity


device = get_test_device()


@pytest.fixture
def xml_with_motor_specs():
  """XML with embedded motor specifications."""
  return """
    <mujoco model="test_robot">
      <worldbody>
        <body name="base">
          <joint name="hip_joint" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="box" size="0.1 0.1 0.1"/>
          <body name="upper">
            <joint name="knee_joint" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="box" size="0.1 0.1 0.2"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="hip_motor" joint="hip_joint" gear="14.5"/>
        <motor name="knee_motor" joint="knee_joint" gear="9"/>
      </actuator>
      <custom>
        <text name="motor_hip_motor" data="motor_spec:unitree_7520_14"/>
        <text name="motor_knee_motor" data="motor_spec:unitree_5020_9"/>
      </custom>
    </mujoco>
    """


@pytest.fixture
def xml_with_battery_spec():
  """XML with embedded battery specification."""
  return """
    <mujoco model="test_robot">
      <worldbody>
        <body name="base">
          <joint name="hip_joint" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="hip_motor" joint="hip_joint" gear="14.5"/>
      </actuator>
      <custom>
        <text name="motor_hip_motor" data="motor_spec:unitree_7520_14"/>
        <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
      </custom>
    </mujoco>
    """


@pytest.fixture
def xml_no_specs():
  """XML without any motor or battery specs."""
  return """
    <mujoco model="test_robot">
      <worldbody>
        <body name="base">
          <joint name="hip_joint" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="hip_motor" joint="hip_joint" gear="14.5"/>
      </actuator>
    </mujoco>
    """


def test_auto_discover_motors_from_xml(xml_with_motor_specs):
  """Test auto-discovery of motors from XML."""
  spec_fn = lambda: mujoco.MjSpec.from_string(xml_with_motor_specs)
  cfg = EntityCfg(spec_fn=spec_fn, auto_discover_motors=True)
  entity = Entity(cfg)

  # Should have auto-created actuators
  assert entity.cfg.articulation is not None
  assert len(entity.cfg.articulation.actuators) > 0

  # Initialize to verify actuators work
  entity, sim = initialize_entity(entity, device)
  assert len(entity._actuators) > 0


def test_auto_discover_multiple_motor_types(xml_with_motor_specs):
  """Test auto-discovery groups motors by type."""
  spec_fn = lambda: mujoco.MjSpec.from_string(xml_with_motor_specs)
  cfg = EntityCfg(spec_fn=spec_fn, auto_discover_motors=True)
  entity = Entity(cfg)

  # Should have 2 actuator groups (7520 and 5020)
  assert entity.cfg.articulation is not None
  assert len(entity.cfg.articulation.actuators) == 2

  # Verify motor specs are correct
  actuator_cfgs = entity.cfg.articulation.actuators
  motor_ids = {cfg.motor_spec.motor_id for cfg in actuator_cfgs}
  assert "unitree_7520_14" in motor_ids
  assert "unitree_5020_9" in motor_ids


def test_auto_discover_motors_disabled(xml_with_motor_specs):
  """Test auto-discovery can be disabled."""
  spec_fn = lambda: mujoco.MjSpec.from_string(xml_with_motor_specs)
  cfg = EntityCfg(spec_fn=spec_fn, auto_discover_motors=False)
  entity = Entity(cfg)

  # Should NOT have created actuators
  assert entity.cfg.articulation is None


def test_auto_discover_battery_from_xml(xml_with_battery_spec):
  """Test auto-discovery of battery from XML."""
  spec_fn = lambda: mujoco.MjSpec.from_string(xml_with_battery_spec)
  scene_cfg = SceneCfg(
    num_envs=1, entities={"robot": EntityCfg(spec_fn=spec_fn)}, auto_battery=True
  )
  scene = Scene(scene_cfg, device=device)

  # Should have auto-created battery manager
  assert scene._battery_manager is not None
  assert scene._battery_manager.cfg.battery_spec.battery_id == "unitree_g1_9ah"


def test_manual_config_takes_precedence(xml_with_motor_specs):
  """Test manual articulation config takes precedence over auto-discovery."""
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg

  spec_fn = lambda: mujoco.MjSpec.from_string(xml_with_motor_specs)

  # Provide explicit articulation config
  manual_actuator = BuiltinPositionActuatorCfg(
    target_names_expr=("hip_joint",), stiffness=100.0, damping=5.0
  )
  articulation_cfg = EntityArticulationInfoCfg(actuators=(manual_actuator,))

  cfg = EntityCfg(
    spec_fn=spec_fn, articulation=articulation_cfg, auto_discover_motors=True
  )
  entity = Entity(cfg)

  # Should use manual config, not auto-discovered motors
  assert entity.cfg.articulation.actuators == (manual_actuator,)
  assert len(entity.cfg.articulation.actuators) == 1


def test_no_specs_in_xml(xml_no_specs):
  """Test graceful handling when no specs are present."""
  spec_fn = lambda: mujoco.MjSpec.from_string(xml_no_specs)

  # Entity with auto-discovery enabled but no specs in XML
  cfg = EntityCfg(spec_fn=spec_fn, auto_discover_motors=True)
  entity = Entity(cfg)

  # Should not error, just have no actuators
  assert entity.cfg.articulation is None

  # Scene with auto-battery enabled but no battery in XML
  scene_cfg = SceneCfg(
    num_envs=1, entities={"robot": EntityCfg(spec_fn=spec_fn)}, auto_battery=True
  )
  scene = Scene(scene_cfg, device=device)

  # Should not error, just have no battery
  assert scene._battery_manager is None


def test_battery_manual_config_precedence(xml_with_battery_spec):
  """Test manual battery config takes precedence over auto-discovery."""
  from mjlab.battery import BatteryManagerCfg
  from mjlab.battery_database import load_battery_spec

  spec_fn = lambda: mujoco.MjSpec.from_string(xml_with_battery_spec)

  # Provide explicit battery config (different from XML)
  manual_battery_cfg = BatteryManagerCfg(
    battery_spec=load_battery_spec("turnigy_6s2p_5000mah"),
    entity_names=("robot",),
    initial_soc=0.8,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    entities={"robot": EntityCfg(spec_fn=spec_fn)},
    battery=manual_battery_cfg,
    auto_battery=True,
  )
  scene = Scene(scene_cfg, device=device)

  # Should use manual config, not auto-discovered battery
  assert scene._battery_manager is not None
  assert scene._battery_manager.cfg.battery_spec.battery_id == "turnigy_6s2p_5000mah"
  assert scene._battery_manager.cfg.initial_soc == 0.8
