"""Motor database for loading motor specifications.

This module provides functionality for loading motor specifications from
various sources including built-in motors, user directories, URLs, and
custom paths.

Example:
    >>> from mjlab.motor_database import load_motor_spec
    >>> motor = load_motor_spec("unitree_7520_14")
    >>> print(f"Continuous torque: {motor.continuous_torque} N⋅m")
"""

from mjlab.motor_database.database import (
  add_motor_database_path,
  get_default_search_paths,
  load_motor_spec,
)
from mjlab.motor_database.motor_spec import MotorSpecification

__all__ = [
  "MotorSpecification",
  "load_motor_spec",
  "add_motor_database_path",
  "get_default_search_paths",
]
