"""XML integration for battery specifications.

This module provides utilities for reading and writing battery_spec references
in MuJoCo XML files using the <custom><text> mechanism.
"""

import mujoco


def write_battery_spec_to_xml(
  spec: mujoco.MjSpec, battery_name: str, battery_id: str
) -> None:
  """Add battery_spec reference to MuJoCo spec as custom text data.

  This stores the battery_spec in a way that:
  - Standard MuJoCo can load without errors
  - Preserves through XML write/read cycles
  - Can be queried programmatically

  Args:
      spec: MuJoCo spec to modify
      battery_name: Name of the battery (e.g., "main_battery")
      battery_id: Battery specification ID

  Example:
      >>> spec = mujoco.MjSpec.from_file("robot.xml")
      >>> write_battery_spec_to_xml(spec, "main_battery", "turnigy_6s2p_5000mah")
      >>> xml_output = spec.to_xml()
      >>> # XML will contain: <text name="battery_main_battery"
      >>> #                         data="battery_spec:turnigy_6s2p_5000mah"/>
  """
  text = spec.add_text()
  text.name = f"battery_{battery_name}"
  text.data = f"battery_spec:{battery_id}"


def parse_battery_specs_from_xml(spec: mujoco.MjSpec) -> dict[str, str]:
  """Parse battery_spec references from MuJoCo spec custom text data.

  Looks for custom text elements with:
  - Name pattern: "battery_{battery_name}"
  - Data pattern: "battery_spec:{battery_id}"

  Args:
      spec: MuJoCo spec to parse

  Returns:
      Dictionary mapping battery_name -> battery_id

  Example:
      >>> spec = mujoco.MjSpec.from_file("robot.xml")
      >>> battery_specs = parse_battery_specs_from_xml(spec)
      >>> # {'main_battery': 'turnigy_6s2p_5000mah',
      >>> #  'aux_battery': 'lifepo4_12s_10ah'}
  """
  battery_specs = {}

  for text in spec.texts:
    # Check if this is a battery_spec entry
    if text.name.startswith("battery_") and text.data.startswith("battery_spec:"):
      battery_name = text.name[8:]  # Remove "battery_" prefix
      battery_id = text.data.split(":", 1)[1]
      battery_specs[battery_name] = battery_id

  return battery_specs


def has_battery_spec(spec: mujoco.MjSpec, battery_name: str) -> bool:
  """Check if a battery has a battery_spec reference.

  Args:
      spec: MuJoCo spec to check
      battery_name: Name of the battery

  Returns:
      True if battery has battery_spec, False otherwise
  """
  text_name = f"battery_{battery_name}"
  for text in spec.texts:
    if text.name == text_name and text.data.startswith("battery_spec:"):
      return True
  return False


def get_battery_spec(spec: mujoco.MjSpec, battery_name: str) -> str | None:
  """Get battery_spec ID for a specific battery.

  Args:
      spec: MuJoCo spec to query
      battery_name: Name of the battery

  Returns:
      Battery ID if found, None otherwise
  """
  text_name = f"battery_{battery_name}"
  for text in spec.texts:
    if text.name == text_name and text.data.startswith("battery_spec:"):
      return text.data.split(":", 1)[1]
  return None


def remove_battery_spec(spec: mujoco.MjSpec, battery_name: str) -> bool:
  """Remove battery_spec reference from a battery.

  Args:
      spec: MuJoCo spec to modify
      battery_name: Name of the battery

  Returns:
      True if battery_spec was removed, False if not found
  """
  text_name = f"battery_{battery_name}"
  for text in spec.texts:
    if text.name == text_name and text.data.startswith("battery_spec:"):
      # Note: MjSpec doesn't have a direct remove method,
      # so we need to clear the data
      text.name = ""
      text.data = ""
      return True
  return False
