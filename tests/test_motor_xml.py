"""Tests for motor_spec XML integration with MuJoCo.

This test suite verifies that motor_spec references can be stored in MuJoCo
XML files using <custom><text> elements and that:
1. Standard MuJoCo can load XMLs with motor_spec (backward compatible)
2. Motor specs preserve through XML write/read roundtrips
3. Helper functions for reading/writing motor specs work correctly
"""

import tempfile
from pathlib import Path

import mujoco
import pytest

from mjlab.motor_database.xml_integration import (
  get_motor_spec,
  has_motor_spec,
  parse_motor_specs_from_xml,
  write_motor_spec_to_xml,
)


def test_mujoco_rejects_custom_attributes():
  """Test that MuJoCo REJECTS custom attributes like motor_spec.

  MuJoCo strictly validates XML schema and does NOT allow custom attributes.
  This test documents that motor_spec must use <custom><text> instead.
  """
  xml_with_custom_attribute = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1" gear="10" motor_spec="test_motor"/>
      </actuator>
    </mujoco>
    """

  # MuJoCo SHOULD reject this - custom attributes not allowed
  with pytest.raises(ValueError, match="unrecognized attribute"):
    mujoco.MjModel.from_xml_string(xml_with_custom_attribute)


def test_mujoco_accepts_custom_text():
  """Test that MuJoCo accepts <custom><text> elements (official way)."""
  xml_with_custom_text = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>

      <custom>
        <text name="motor_motor1" data="motor_spec:test_motor"/>
      </custom>
    </mujoco>
    """

  # This should work - custom text is official MuJoCo feature
  model = mujoco.MjModel.from_xml_string(xml_with_custom_text)
  assert model is not None
  assert model.nu == 1  # One actuator


def test_write_motor_spec_to_xml():
  """Test writing motor_spec to MuJoCo spec."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)

  # Write motor spec
  write_motor_spec_to_xml(spec, "motor1", "unitree_7520_14")

  # Check it was added
  texts = list(spec.texts)
  assert len(texts) == 1
  assert texts[0].name == "motor_motor1"
  assert texts[0].data == "motor_spec:unitree_7520_14"


def test_parse_motor_specs_from_xml():
  """Test parsing motor_specs from XML."""
  xml_with_motors = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
          <joint name="joint2" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="left_hip" joint="joint1"/>
        <motor name="right_hip" joint="joint2"/>
      </actuator>

      <custom>
        <text name="motor_left_hip" data="motor_spec:unitree_7520_14"/>
        <text name="motor_right_hip" data="motor_spec:unitree_7520_14"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_with_motors)
  motor_specs = parse_motor_specs_from_xml(spec)

  assert len(motor_specs) == 2
  assert motor_specs["left_hip"] == "unitree_7520_14"
  assert motor_specs["right_hip"] == "unitree_7520_14"


def test_motor_spec_xml_roundtrip():
  """Test that motor_spec survives XML write/read roundtrip."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>
    </mujoco>
    """

  # Load spec and add motor_spec
  spec = mujoco.MjSpec.from_string(xml_template)
  write_motor_spec_to_xml(spec, "motor1", "test_motor")

  # Write to file
  with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
    temp_path = Path(f.name)
    xml_output = spec.to_xml()
    f.write(xml_output)

  try:
    # Read back
    spec2 = mujoco.MjSpec.from_file(str(temp_path))
    motor_specs = parse_motor_specs_from_xml(spec2)

    # Verify motor_spec preserved
    assert "motor1" in motor_specs
    assert motor_specs["motor1"] == "test_motor"
  finally:
    temp_path.unlink()


def test_has_motor_spec():
  """Test checking if actuator has motor_spec."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
          <joint name="joint2" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
        <motor name="motor2" joint="joint2"/>
      </actuator>

      <custom>
        <text name="motor_motor1" data="motor_spec:test_motor"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)

  assert has_motor_spec(spec, "motor1") is True
  assert has_motor_spec(spec, "motor2") is False
  assert has_motor_spec(spec, "nonexistent") is False


def test_get_motor_spec():
  """Test getting motor_spec ID for an actuator."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>

      <custom>
        <text name="motor_motor1" data="motor_spec:unitree_7520_14"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)

  assert get_motor_spec(spec, "motor1") == "unitree_7520_14"
  assert get_motor_spec(spec, "motor2") is None


def test_multiple_motors_in_xml():
  """Test XML with multiple actuators having different motor specs."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
          <joint name="joint2" type="hinge"/>
          <joint name="joint3" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="hip_motor" joint="joint1"/>
        <motor name="knee_motor" joint="joint2"/>
        <motor name="ankle_motor" joint="joint3"/>
      </actuator>

      <custom>
        <text name="motor_hip_motor" data="motor_spec:unitree_7520_14"/>
        <text name="motor_knee_motor" data="motor_spec:unitree_5020_9"/>
        <text name="motor_ankle_motor" data="motor_spec:unitree_5020_9"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)
  motor_specs = parse_motor_specs_from_xml(spec)

  assert len(motor_specs) == 3
  assert motor_specs["hip_motor"] == "unitree_7520_14"
  assert motor_specs["knee_motor"] == "unitree_5020_9"
  assert motor_specs["ankle_motor"] == "unitree_5020_9"


def test_mixed_actuators_with_and_without_motor_spec():
  """Test XML with some actuators having motor_spec and others not."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
          <joint name="joint2" type="hinge"/>
          <joint name="joint3" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
        <motor name="motor2" joint="joint2"/>
        <position name="pos1" joint="joint3" kp="100"/>
      </actuator>

      <custom>
        <text name="motor_motor1" data="motor_spec:test_motor"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)
  motor_specs = parse_motor_specs_from_xml(spec)

  # Only motor1 has motor_spec
  assert len(motor_specs) == 1
  assert motor_specs["motor1"] == "test_motor"


def test_motor_spec_with_special_characters():
  """Test motor_spec IDs with underscores, numbers, etc."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>

      <custom>
        <text name="motor_motor1"
              data="motor_spec:unitree_7520_14_calibrated_v2"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)
  assert get_motor_spec(spec, "motor1") == "unitree_7520_14_calibrated_v2"


def test_write_and_parse_integration():
  """Test writing motor specs and parsing them back."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
          <joint name="joint2" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="left_motor" joint="joint1"/>
        <motor name="right_motor" joint="joint2"/>
      </actuator>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)

  # Write multiple motor specs
  write_motor_spec_to_xml(spec, "left_motor", "unitree_7520_14")
  write_motor_spec_to_xml(spec, "right_motor", "unitree_5020_9")

  # Parse them back
  motor_specs = parse_motor_specs_from_xml(spec)

  assert len(motor_specs) == 2
  assert motor_specs["left_motor"] == "unitree_7520_14"
  assert motor_specs["right_motor"] == "unitree_5020_9"


def test_mujoco_model_compile_with_motor_spec():
  """Test that model compiles successfully with motor_spec in XML."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1" gear="10"/>
      </actuator>

      <custom>
        <text name="motor_motor1" data="motor_spec:test_motor"/>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)

  # Should compile successfully
  model = spec.compile()
  assert model is not None
  assert model.nu == 1


def test_empty_custom_section():
  """Test parsing XML with empty custom section."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>

      <custom>
      </custom>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)
  motor_specs = parse_motor_specs_from_xml(spec)

  assert len(motor_specs) == 0


def test_no_custom_section():
  """Test parsing XML without custom section."""
  xml_template = """
    <mujoco model="test">
      <worldbody>
        <body name="body1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint name="joint1" type="hinge"/>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1"/>
      </actuator>
    </mujoco>
    """

  spec = mujoco.MjSpec.from_string(xml_template)
  motor_specs = parse_motor_specs_from_xml(spec)

  assert len(motor_specs) == 0


def test_g1_robot_with_motor_specs():
  """Test adding motor specs to real G1 Unitree robot model."""
  # Simplified G1 XML with key actuators
  g1_xml = """
    <mujoco model="g1_29dof">
      <compiler angle="radian"/>
      <option timestep=".004"/>

      <worldbody>
        <body name="pelvis" pos="0 0 0.793">
          <freejoint name="floating_base_joint"/>
          <geom type="sphere" size="0.07"/>

          <body name="left_hip_pitch_link">
            <joint name="left_hip_pitch_joint" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.05"/>

            <body name="left_hip_roll_link">
              <joint name="left_hip_roll_joint" type="hinge" axis="1 0 0"/>
              <geom type="sphere" size="0.05"/>

              <body name="left_hip_yaw_link">
                <joint name="left_hip_yaw_joint" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05"/>

                <body name="left_knee_link">
                  <joint name="left_knee_joint" type="hinge" axis="0 1 0"/>
                  <geom type="sphere" size="0.04"/>

                  <body name="left_ankle_pitch_link">
                    <joint name="left_ankle_pitch_joint" type="hinge"
                           axis="0 1 0"/>
                    <geom type="sphere" size="0.03"/>
                  </body>
                </body>
              </body>
            </body>
          </body>

          <body name="left_shoulder_pitch_link">
            <joint name="left_shoulder_pitch_joint" type="hinge"
                   axis="0 1 0"/>
            <geom type="sphere" size="0.04"/>

            <body name="left_elbow_link">
              <joint name="left_elbow_joint" type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.03"/>
            </body>
          </body>
        </body>
      </worldbody>

      <actuator>
        <position name="left_hip_pitch_joint" joint="left_hip_pitch_joint"
                  kp="75"/>
        <position name="left_hip_roll_joint" joint="left_hip_roll_joint"
                  kp="75"/>
        <position name="left_hip_yaw_joint" joint="left_hip_yaw_joint"
                  kp="75"/>
        <position name="left_knee_joint" joint="left_knee_joint" kp="75"/>
        <position name="left_ankle_pitch_joint"
                  joint="left_ankle_pitch_joint" kp="20"/>
        <position name="left_shoulder_pitch_joint"
                  joint="left_shoulder_pitch_joint" kp="75"/>
        <position name="left_elbow_joint" joint="left_elbow_joint" kp="75"/>
      </actuator>
    </mujoco>
    """

  # Load G1 spec
  spec = mujoco.MjSpec.from_string(g1_xml)

  # Add motor specs for different joint types
  # Hip joints: High-torque motors (88 N⋅m)
  write_motor_spec_to_xml(spec, "left_hip_pitch_joint", "unitree_7520_14")
  write_motor_spec_to_xml(spec, "left_hip_roll_joint", "unitree_7520_14")
  write_motor_spec_to_xml(spec, "left_hip_yaw_joint", "unitree_7520_14")

  # Knee joint: High-torque motor (88 N⋅m)
  write_motor_spec_to_xml(spec, "left_knee_joint", "unitree_7520_14")

  # Ankle joint: Mid-range motor (25 N⋅m)
  write_motor_spec_to_xml(spec, "left_ankle_pitch_joint", "unitree_5020_9")

  # Arm joints: Mid-range motors (25 N⋅m)
  write_motor_spec_to_xml(spec, "left_shoulder_pitch_joint", "unitree_5020_9")
  write_motor_spec_to_xml(spec, "left_elbow_joint", "unitree_5020_9")

  # Parse back motor specs
  motor_specs = parse_motor_specs_from_xml(spec)

  # Verify all 7 motor specs were added
  assert len(motor_specs) == 7
  assert motor_specs["left_hip_pitch_joint"] == "unitree_7520_14"
  assert motor_specs["left_hip_roll_joint"] == "unitree_7520_14"
  assert motor_specs["left_hip_yaw_joint"] == "unitree_7520_14"
  assert motor_specs["left_knee_joint"] == "unitree_7520_14"
  assert motor_specs["left_ankle_pitch_joint"] == "unitree_5020_9"
  assert motor_specs["left_shoulder_pitch_joint"] == "unitree_5020_9"
  assert motor_specs["left_elbow_joint"] == "unitree_5020_9"

  # Verify model compiles with motor specs
  model = spec.compile()
  assert model is not None
  assert model.nu == 7  # 7 actuators

  # Write to XML and verify roundtrip
  xml_output = spec.to_xml()
  assert "motor_left_hip_pitch_joint" in xml_output
  assert "motor_spec:unitree_7520_14" in xml_output
  assert "motor_spec:unitree_5020_9" in xml_output

  # Load from XML string and verify motor specs preserved
  spec2 = mujoco.MjSpec.from_string(xml_output)
  motor_specs2 = parse_motor_specs_from_xml(spec2)
  assert motor_specs == motor_specs2


if __name__ == "__main__":
  # Run tests manually for quick verification
  print("Testing MuJoCo motor_spec XML integration...\n")

  test_mujoco_rejects_custom_attributes()
  print("✓ MuJoCo rejects custom attributes (as expected)")

  test_mujoco_accepts_custom_text()
  print("✓ MuJoCo accepts <custom><text> elements")

  test_write_motor_spec_to_xml()
  print("✓ write_motor_spec_to_xml() works")

  test_parse_motor_specs_from_xml()
  print("✓ parse_motor_specs_from_xml() works")

  test_motor_spec_xml_roundtrip()
  print("✓ motor_spec survives XML roundtrip")

  test_has_motor_spec()
  print("✓ has_motor_spec() works")

  test_get_motor_spec()
  print("✓ get_motor_spec() works")

  test_multiple_motors_in_xml()
  print("✓ Multiple motors with different specs work")

  test_mixed_actuators_with_and_without_motor_spec()
  print("✓ Mixed actuators (with/without motor_spec) work")

  test_motor_spec_with_special_characters()
  print("✓ motor_spec with special characters works")

  test_write_and_parse_integration()
  print("✓ Write and parse integration works")

  test_mujoco_model_compile_with_motor_spec()
  print("✓ Model compiles with motor_spec")

  test_empty_custom_section()
  print("✓ Empty custom section works")

  test_no_custom_section()
  print("✓ No custom section works")

  test_g1_robot_with_motor_specs()
  print("✓ G1 robot with motor specs works")

  print("\n✅ All motor_spec XML integration tests passed!")
