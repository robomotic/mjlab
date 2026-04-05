# Motor Database

The motor database provides a flexible system for loading motor specifications with automatic fallback to remote repositories.

## Quick Start

```python
from mjlab.motor_database import load_motor_spec

# Load from built-in database
motor = load_motor_spec("unitree_7520_14")

# Load from GitHub (automatic fallback)
motor = load_motor_spec("faulhaber_2264w024bp4")

# Load from explicit URL
motor = load_motor_spec(url="https://example.com/motor.json")

# Load from file
motor = load_motor_spec(file="/path/to/motor.json")
```

## Search Path Priority

When loading a motor by ID, the system searches in this order:

1. **User directory**: `~/.mjlab/motors/`
2. **Project directory**: `./motors/`
3. **Environment variable**: `$MJLAB_MOTOR_PATH` (colon-separated)
4. **Programmatically added paths**: via `add_motor_database_path()`
5. **Built-in database**: Shipped with mjlab
6. **Remote repositories**: GitHub with automatic caching

## Remote Repository Support

### Automatic GitHub Fallback

Motor specifications are automatically downloaded from community repositories:

```python
# If not found locally, automatically fetches from GitHub
motor = load_motor_spec("faulhaber_2264w024bp4")
```

**URL Pattern:**
```
https://raw.githubusercontent.com/robomotic/mujoco-motors/master/motor_assets/{manufacturer}/{motor_id}.json
```

**Example:**
- Motor ID: `faulhaber_2264w024bp4`
- Manufacturer: `faulhaber` (extracted from motor_id)
- URL: `https://raw.githubusercontent.com/robomotic/mujoco-motors/master/motor_assets/faulhaber/faulhaber_2264w024bp4.json`

### Remote Repository Configuration

Default repository is configured in `database.py`:
```python
REMOTE_MOTOR_REPOSITORIES = [
  "https://raw.githubusercontent.com/robomotic/mujoco-motors/master/motor_assets"
]
```

You can add custom remote repositories by modifying this list.

## Cache Management

### Cache Location

Downloaded motors are cached at: `~/.mjlab/cache/motors/`

Files are named using MD5 hashes of the source URL.

### Viewing Cache

```bash
# List cached motors
ls -lh ~/.mjlab/cache/motors/

# View a cached motor
cat ~/.mjlab/cache/motors/<hash>.json | python -m json.tool
```

### Clearing Cache

```bash
# Remove all cached motors
rm -rf ~/.mjlab/cache/motors/

# Clear cache and force re-download
rm -rf ~/.mjlab/cache/motors/ && python -c "from mjlab.motor_database import load_motor_spec; load_motor_spec('faulhaber_2264w024bp4')"
```

### When to Clear Cache

- Motor specification updated on GitHub
- Corrupted cache files
- Testing motor specification changes
- Disk space cleanup

## Custom Search Paths

### Environment Variable

```bash
# Single path
export MJLAB_MOTOR_PATH="/path/to/motors"

# Multiple paths (colon-separated)
export MJLAB_MOTOR_PATH="/path/to/motors:/another/path"

# Include cloned GitHub repo for offline access
export MJLAB_MOTOR_PATH="$HOME/repos/mujoco-motors/motor_assets:$MJLAB_MOTOR_PATH"
```

### Programmatic Configuration

```python
from mjlab.motor_database import add_motor_database_path

# Add custom search path
add_motor_database_path("/path/to/custom/motors")

# Load motor from added path
motor = load_motor_spec("custom_motor_id")
```

## Motor Specification Format

Motors are defined in JSON files with required and optional fields:

### Required Fields (14)

```json
{
  "motor_id": "example_motor",
  "manufacturer": "Example Corp",
  "model": "EX-100",
  "voltage_range": [0.0, 24.0],
  "resistance": 0.5,
  "inductance": 0.0002,
  "motor_constant_kt": 0.12,
  "motor_constant_ke": 0.12,
  "peak_torque": 50.0,
  "no_load_speed": 100.0,
  "thermal_resistance": 5.0,
  "thermal_time_constant": 300.0,
  "max_winding_temperature": 120.0
}
```

### Optional Fields (19)

All optional fields have sensible defaults:
- `number_of_pole_pairs` (int, default: None) - For commutation frequency
- `commutation` (str, default: None) - Sensor type
- `max_speed` (float, default: None) - Mechanical bearing limit
- `weight` (float, default: 0.0) - Motor mass in kg
- `friction_static` (float, default: 0.0) - Static friction torque
- `friction_dynamic` (float, default: 0.0) - Dynamic friction coefficient
- `gear_ratio` (float, default: 1.0)
- `reflected_inertia` (float, default: 0.0)
- `rotation_angle_range` (tuple, default: [-π, π])
- `stall_torque` (float, default: 10.0)
- `continuous_torque` (float, default: 10.0)
- `no_load_current` (float, default: 0.5)
- `stall_current` (float, default: 10.0)
- `operating_current` (float, default: 3.0)
- `ambient_temperature` (float, default: 25.0)
- `encoder_resolution` (int, default: 2048)
- `encoder_type` (str, default: "incremental")
- `feedback_sensors` (list, default: ["position", "velocity"])
- `protocol` (str, default: "PWM")
- `protocol_params` (dict, default: {})
- `step_file` (str, default: None) - Path to STEP CAD file
- `stl_file` (str, default: None) - Path to STL mesh file

See [motor_spec.py](motor_spec.py) for complete field documentation.

## Usage in MuJoCo XML

Reference motors in XML using custom text elements:

```xml
<mujoco model="robot">
  <actuator>
    <motor name="left_hip" joint="hip_joint" gear="14.5"/>
  </actuator>

  <custom>
    <!-- Motor specification (loaded automatically) -->
    <text name="motor_left_hip" data="motor_spec:faulhaber_2264w024bp4"/>
  </custom>
</mujoco>
```

The motor will be automatically loaded (from local database or GitHub) when the model is loaded.

## Examples

### Running Examples Locally

**Basic motor database usage:**
```bash
# From repository root
uv run python examples/motor_database_basic.py
```

This example demonstrates:
- Loading motors from built-in database
- Loading motors from remote GitHub repository (automatic fallback)
- Accessing motor properties (torque, speed, resistance, etc.)
- Adding custom search paths

**Interactive CartPole with electrical motors:**
```bash
# Run CartPole with sinusoidal torque commands
uv run play Mjlab-Cartpole-Constant-Rotation --agent sin --viewer viser

# Try different frequencies and amplitudes
uv run play Mjlab-Cartpole-Constant-Rotation --agent sin --sin_frequency 2.0 --sin_amplitude 20.0 --viewer viser

# Compare with zero agent (no torque)
uv run play Mjlab-Cartpole-Constant-Rotation --agent zero --viewer viser

# Record video (saved to logs/play/{task}/{agent}/{timestamp}/videos/play/)
uv run play Mjlab-Cartpole-Constant-Rotation --agent sin --viewer viser --video --video_length 500
```

This demonstrates:
- Electrical motor actuators with realistic voltage/current/power
- Real-time metrics visualization (voltage, current, power, temperature)
- Direct torque control (no PD feedback)
- Battery state tracking (SOC, voltage sag, thermal heating)
- Video recording with custom length and resolution

### Jupyter Notebooks (Colab Compatible)

**Tutorial 1: Motor Physics Fundamentals**
- Notebook: [`notebooks/electrical/01_intro.ipynb`](../../../notebooks/electrical/01_intro.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robomotic/mjlab/blob/feature/motor-database-extension/notebooks/electrical/01_intro.ipynb)
- Topics:
  - Electrical motor modeling (voltage, current, back-EMF)
  - RL circuit dynamics and inductance effects
  - Thermal modeling (I²R heating, temperature limits)
  - Battery discharge and voltage sag
  - Real-time metrics visualization

**Tutorial 2: Motor Sizing with CartPole**
- Notebook: [`notebooks/electrical/02_intro.ipynb`](../../../notebooks/electrical/02_intro.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robomotic/mjlab/blob/feature/motor-database-extension/notebooks/electrical/02_intro.ipynb)
- Topics:
  - Comparing underpowered vs well-sized vs overpowered motors
  - Sinusoidal torque patterns for cyclic load testing
  - Power consumption analysis
  - The 2-4× sizing rule (motors at 25-50% of peak torque)
  - Motor saturation and thermal limits

**Humanoid Motor Demo (Automatic API)**
- Notebook: [`notebooks/humanoid_motor_demo_easy.ipynb`](../../../notebooks/humanoid_motor_demo_easy.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robomotic/mjlab/blob/feature/motor-database-extension/notebooks/humanoid_motor_demo_easy.ipynb)
- Topics:
  - Unitree G1 humanoid with automatic motor/battery discovery
  - Arm movement simulation with real motor specs
  - Automatic current calculation from torque
  - Battery discharge tracking (SOC, voltage, temperature)
  - 90% less code than manual approach

**Running Locally:**
```bash
# Install Jupyter if not already installed
uv add --dev jupyter matplotlib ipywidgets

# Run Tutorial 1
uv run jupyter notebook notebooks/electrical/01_intro.ipynb

# Run Tutorial 2
uv run jupyter notebook notebooks/electrical/02_intro.ipynb

# Run Humanoid Demo
uv run jupyter notebook notebooks/humanoid_motor_demo_easy.ipynb
```

### Key Features Demonstrated

All examples showcase:
- ✅ **Auto-discovery** of motors from XML custom tags
- ✅ **Automatic** electrical calculations (voltage, current, power)
- ✅ **Real-time** battery tracking (SOC, voltage sag, thermal heating)
- ✅ **Realistic** motor physics (saturation, back-EMF, inductance)
- ✅ **Zero manual** physics - just load and step!

See [examples/motor_database_basic.py](../../../examples/motor_database_basic.py) for programmatic motor loading examples.

## Community Repository

Community-maintained motor specifications: https://github.com/robomotic/mujoco-motors

Structure:
```
motor_assets/
├── faulhaber/
│   ├── faulhaber_2264w024bp4.json
│   └── ...
├── dynamixel/
│   ├── dynamixel_xl330_m288_t.json
│   └── ...
├── unitree/
│   ├── unitree_7520_14.json
│   └── ...
└── ...
```

To contribute motors, open a pull request to the repository.
