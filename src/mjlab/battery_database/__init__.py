"""Battery database module for mjlab.

This module provides battery specifications for common rechargeable batteries
used in robotics (LiPo, LiFePO4, Li-ion) with flexible loading from various
sources.

Public API:
    - BatterySpecification: Battery specification dataclass
    - load_battery_spec: Load battery from database by ID, file, or URL
    - get_default_search_paths: Get default battery database search paths
    - add_battery_database_path: Add a custom battery database path
    - write_battery_spec_to_xml: Write battery spec to MuJoCo XML
    - parse_battery_specs_from_xml: Parse battery specs from MuJoCo XML
"""

from mjlab.battery_database.battery_spec import BatterySpecification
from mjlab.battery_database.database import (
  add_battery_database_path,
  get_default_search_paths,
  load_battery_spec,
)
from mjlab.battery_database.xml_integration import (
  get_battery_spec,
  has_battery_spec,
  parse_battery_specs_from_xml,
  remove_battery_spec,
  write_battery_spec_to_xml,
)

__all__ = [
  "BatterySpecification",
  "load_battery_spec",
  "get_default_search_paths",
  "add_battery_database_path",
  "write_battery_spec_to_xml",
  "parse_battery_specs_from_xml",
  "get_battery_spec",
  "has_battery_spec",
  "remove_battery_spec",
]
