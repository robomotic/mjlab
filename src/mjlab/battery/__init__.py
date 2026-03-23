"""Battery module for mjlab.

Provides battery management and power distribution for scene-level simulation.

Public API:
    - BatteryManagerCfg: Configuration for battery manager
    - BatteryManager: Scene-level battery state tracking and power aggregation
"""

from mjlab.battery.battery_manager import BatteryManager, BatteryManagerCfg

__all__ = [
  "BatteryManager",
  "BatteryManagerCfg",
]
