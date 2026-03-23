"""XML integration for motor specifications.

This module provides utilities for reading and writing motor_spec references
in MuJoCo XML files using the <custom><text> mechanism.
"""

import mujoco


def write_motor_spec_to_xml(
  spec: mujoco.MjSpec, actuator_name: str, motor_id: str
) -> None:
  """Add motor_spec reference to MuJoCo spec as custom text data.

  This stores the motor_spec in a way that:
  - Standard MuJoCo can load without errors
  - Preserves through XML write/read cycles
  - Can be queried programmatically

  Args:
      spec: MuJoCo spec to modify
      actuator_name: Name of the actuator
      motor_id: Motor specification ID

  Example:
      >>> spec = mujoco.MjSpec.from_file("robot.xml")
      >>> write_motor_spec_to_xml(spec, "left_hip_motor", "unitree_7520_14")
      >>> xml_output = spec.to_xml()
      >>> # XML will contain: <text name="motor_left_hip_motor"
      >>> #                         data="motor_spec:unitree_7520_14"/>
  """
  text = spec.add_text()
  text.name = f"motor_{actuator_name}"
  text.data = f"motor_spec:{motor_id}"


def parse_motor_specs_from_xml(spec: mujoco.MjSpec) -> dict[str, str]:
  """Parse motor_spec references from MuJoCo spec custom text data.

  Looks for custom text elements with:
  - Name pattern: "motor_{actuator_name}"
  - Data pattern: "motor_spec:{motor_id}"

  Args:
      spec: MuJoCo spec to parse

  Returns:
      Dictionary mapping actuator_name -> motor_id

  Example:
      >>> spec = mujoco.MjSpec.from_file("robot.xml")
      >>> motor_specs = parse_motor_specs_from_xml(spec)
      >>> # {'left_hip_motor': 'unitree_7520_14',
      >>> #  'right_hip_motor': 'unitree_7520_14'}
  """
  motor_specs = {}

  for text in spec.texts:
    # Check if this is a motor_spec entry
    if text.name.startswith("motor_") and text.data.startswith("motor_spec:"):
      actuator_name = text.name[6:]  # Remove "motor_" prefix
      motor_id = text.data.split(":", 1)[1]
      motor_specs[actuator_name] = motor_id

  return motor_specs


def has_motor_spec(spec: mujoco.MjSpec, actuator_name: str) -> bool:
  """Check if an actuator has a motor_spec reference.

  Args:
      spec: MuJoCo spec to check
      actuator_name: Name of the actuator

  Returns:
      True if actuator has motor_spec, False otherwise
  """
  text_name = f"motor_{actuator_name}"
  for text in spec.texts:
    if text.name == text_name and text.data.startswith("motor_spec:"):
      return True
  return False


def get_motor_spec(spec: mujoco.MjSpec, actuator_name: str) -> str | None:
  """Get motor_spec ID for a specific actuator.

  Args:
      spec: MuJoCo spec to query
      actuator_name: Name of the actuator

  Returns:
      Motor ID if found, None otherwise
  """
  text_name = f"motor_{actuator_name}"
  for text in spec.texts:
    if text.name == text_name and text.data.startswith("motor_spec:"):
      return text.data.split(":", 1)[1]
  return None


def remove_motor_spec(spec: mujoco.MjSpec, actuator_name: str) -> bool:
  """Remove motor_spec reference from an actuator.

  Args:
      spec: MuJoCo spec to modify
      actuator_name: Name of the actuator

  Returns:
      True if motor_spec was removed, False if not found
  """
  text_name = f"motor_{actuator_name}"
  for text in spec.texts:
    if text.name == text_name and text.data.startswith("motor_spec:"):
      # Note: MjSpec doesn't have a direct remove method,
      # so we need to clear the data
      text.name = ""
      text.data = ""
      return True
  return False
