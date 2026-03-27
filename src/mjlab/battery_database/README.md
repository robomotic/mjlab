# Battery Database

The battery database provides a flexible system for loading battery specifications with automatic fallback to remote repositories.

## Quick Start

```python
from mjlab.battery_database import load_battery_spec

# Load from built-in database
battery = load_battery_spec("turnigy_6s2p_5000mah")

# Load from GitHub (automatic fallback)
battery = load_battery_spec("unitree_g1_9ah")

# Load from explicit URL
battery = load_battery_spec(url="https://example.com/battery.json")

# Load from file
battery = load_battery_spec(file="/path/to/battery.json")
```

## Search Path Priority

When loading a battery by ID, the system searches in this order:

1. **User directory**: `~/.mjlab/batteries/`
2. **Project directory**: `./batteries/`
3. **Environment variable**: `$MJLAB_BATTERY_PATH` (colon-separated)
4. **Programmatically added paths**: via `add_battery_database_path()`
5. **Built-in database**: Shipped with mjlab
6. **Remote repositories**: GitHub with automatic caching

## Remote Repository Support

### Automatic GitHub Fallback

Battery specifications are automatically downloaded from community repositories:

```python
# If not found locally, automatically fetches from GitHub
battery = load_battery_spec("unitree_g1_9ah")
```

**URL Pattern:**
```
https://raw.githubusercontent.com/robomotic/mujoco-batteries/master/battery_assets/{manufacturer}/{battery_id}.json
```

**Example:**
- Battery ID: `unitree_g1_9ah`
- Manufacturer: `unitree` (extracted from battery_id)
- URL: `https://raw.githubusercontent.com/robomotic/mujoco-batteries/master/battery_assets/unitree/unitree_g1_9ah.json`

### Remote Repository Configuration

Default repository is configured in `database.py`:
```python
REMOTE_BATTERY_REPOSITORIES = [
  "https://raw.githubusercontent.com/robomotic/mujoco-batteries/master/battery_assets"
]
```

You can add custom remote repositories by modifying this list.

## Cache Management

### Cache Location

Downloaded batteries are cached at: `~/.mjlab/cache/batteries/`

Files are named using MD5 hashes of the source URL.

### Viewing Cache

```bash
# List cached batteries
ls -lh ~/.mjlab/cache/batteries/

# View a cached battery
cat ~/.mjlab/cache/batteries/<hash>.json | python -m json.tool
```

### Clearing Cache

```bash
# Remove all cached batteries
rm -rf ~/.mjlab/cache/batteries/

# Clear cache and force re-download
rm -rf ~/.mjlab/cache/batteries/ && python -c "from mjlab.battery_database import load_battery_spec; load_battery_spec('unitree_g1_9ah')"
```

### When to Clear Cache

- Battery specification updated on GitHub
- Corrupted cache files
- Testing battery specification changes
- Disk space cleanup

## Custom Search Paths

### Environment Variable

```bash
# Single path
export MJLAB_BATTERY_PATH="/path/to/batteries"

# Multiple paths (colon-separated)
export MJLAB_BATTERY_PATH="/path/to/batteries:/another/path"

# Include cloned GitHub repo for offline access
export MJLAB_BATTERY_PATH="$HOME/repos/mujoco-batteries/battery_assets:$MJLAB_BATTERY_PATH"
```

### Programmatic Configuration

```python
from mjlab.battery_database import add_battery_database_path

# Add custom search path
add_battery_database_path("/path/to/custom/batteries")

# Load battery from added path
battery = load_battery_spec("custom_battery_id")
```

## Battery Specification Format

Batteries are defined in JSON files with required and optional fields:

### Required Fields

```json
{
  "battery_id": "example_battery",
  "manufacturer": "Example Corp",
  "model": "EX-100",
  "chemistry": "LiPo",
  "cells_series": 6,
  "cells_parallel": 2,
  "nominal_cell_voltage": 3.7,
  "max_cell_voltage": 4.2,
  "min_cell_voltage": 3.0,
  "capacity_ah": 5.0,
  "internal_resistance": 0.008,
  "max_continuous_current": 100.0
}
```

### Optional Fields

All optional fields have sensible defaults:
- `min_operating_voltage` (float) - Minimum safe voltage
- `energy_wh` (float, computed if not provided) - Total energy capacity
- `internal_resistance_temp_coeff` (float, default: 0.0) - Temperature coefficient
- `internal_resistance_soc_curve` (list, default: [[0.0, 2.0], [0.5, 1.0], [1.0, 1.0]]) - Resistance vs SOC
- `max_burst_current` (float, default: 2x continuous) - Peak current
- `burst_duration` (float, default: 10.0) - Burst time in seconds
- `c_rating_continuous` (float, computed) - Continuous discharge rate
- `c_rating_burst` (float, computed) - Burst discharge rate
- `ocv_curve` (list, default: linear) - Open circuit voltage vs SOC
- `thermal_capacity` (float, default: 1000.0) - Heat capacity J/K
- `thermal_resistance` (float, default: 10.0) - Thermal resistance K/W
- `max_temperature` (float, default: 60.0) - Max operating temperature °C
- `min_temperature` (float, default: 0.0) - Min operating temperature °C
- `ambient_temperature` (float, default: 25.0) - Ambient temperature °C
- `min_soc` (float, default: 0.2) - Minimum state of charge
- `max_soc` (float, default: 1.0) - Maximum state of charge
- `mass_kg` (float, default: 0.0) - Battery mass
- `volume_liters` (float, default: 0.0) - Battery volume
- `cell_balance_tolerance` (float, default: 0.05) - Cell voltage tolerance

See [battery_spec.py](battery_spec.py) for complete field documentation.

## Usage in MuJoCo XML

Reference batteries in XML using custom text elements:

```xml
<mujoco model="robot">
  <custom>
    <!-- Battery specification (loaded automatically) -->
    <text name="battery_main" data="battery_spec:unitree_g1_9ah"/>
  </custom>
</mujoco>
```

The battery will be automatically loaded (from local database or GitHub) when the model is loaded.

## Community Repository

Community-maintained battery specifications: https://github.com/robomotic/mujoco-batteries

Structure:
```
battery_assets/
├── turnigy/
│   ├── turnigy_6s2p_5000mah.json
│   └── ...
├── unitree/
│   ├── unitree_g1_9ah.json
│   └── ...
├── samsung/
│   ├── samsung_21700_50e.json
│   └── ...
└── ...
```

To contribute batteries, open a pull request to the repository.

## Examples

### Basic Battery Loading

```python
from mjlab.battery_database import load_battery_spec

# Load from built-in database
battery = load_battery_spec("turnigy_6s2p_5000mah")
print(f"Loaded: {battery.battery_id}")
print(f"Chemistry: {battery.chemistry}")
print(f"Capacity: {battery.capacity_ah} Ah")
print(f"Nominal voltage: {battery.nominal_voltage} V")
print(f"Max continuous current: {battery.max_continuous_current} A")

# Load from remote repository (automatic)
battery = load_battery_spec("unitree_g1_9ah")
print(f"Loaded from GitHub: {battery.manufacturer} {battery.model}")

# Access computed properties
print(f"Total energy: {battery.energy_wh} Wh")
print(f"C-rating: {battery.c_rating_continuous}C continuous")
```

### Using with Battery Manager

```python
from mjlab.battery_database import load_battery_spec
from mjlab.battery import BatteryManagerCfg

# Load battery spec
battery_spec = load_battery_spec("unitree_g1_9ah")

# Configure battery manager
battery_cfg = BatteryManagerCfg(
  battery_spec=battery_spec,
  entity_names=("robot",),
  initial_soc=1.0,
  enable_voltage_feedback=True,
)
```

### Custom Battery Path

```python
from mjlab.battery_database import add_battery_database_path, load_battery_spec

# Add custom directory
add_battery_database_path("/my/custom/batteries")

# Load from custom path (searched before built-in)
battery = load_battery_spec("my_custom_battery")
```

### Load from URL

```python
from mjlab.battery_database import load_battery_spec

# Load directly from URL (cached automatically)
battery = load_battery_spec(
  url="https://example.com/batteries/custom_battery.json"
)
```

## See Also

- [Motor Database](../motor_database/README.md) - Similar system for loading motor specs
- [Battery Manager](../battery/battery_manager.py) - Runtime battery state tracking
- [Battery Specification](battery_spec.py) - Complete spec field documentation
