"""Motor database for loading motor specifications.

This module provides functionality for loading motor specifications from
various sources including built-in motors, user directories, URLs, and
custom paths. It also provides XML integration for reading and writing
motor_spec references in MuJoCo XML files.

Example:
    >>> from mjlab.motor_database import load_motor_spec
    >>> motor = load_motor_spec("unitree_7520_14")
    >>> print(f"Continuous torque: {motor.continuous_torque} N⋅m")

    >>> # XML integration
    >>> from mjlab.motor_database import write_motor_spec_to_xml
    >>> import mujoco
    >>> spec = mujoco.MjSpec.from_file("robot.xml")
    >>> write_motor_spec_to_xml(spec, "left_hip_motor", "unitree_7520_14")
"""

from mjlab.motor_database.database import (
  add_motor_database_path,
  get_default_search_paths,
  load_motor_spec,
)
from mjlab.motor_database.motor_spec import MotorSpecification
from mjlab.motor_database.xml_integration import (
  get_motor_spec,
  has_motor_spec,
  parse_motor_specs_from_xml,
  remove_motor_spec,
  write_motor_spec_to_xml,
)

__all__ = [
  "MotorSpecification",
  "load_motor_spec",
  "add_motor_database_path",
  "get_default_search_paths",
  "write_motor_spec_to_xml",
  "parse_motor_specs_from_xml",
  "has_motor_spec",
  "get_motor_spec",
  "remove_motor_spec",
]
