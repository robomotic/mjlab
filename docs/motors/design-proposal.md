# Motor Database with Electrical Characteristics - Design Proposal

**Status:** Proposal
**Date:** 2026-03-20

## Objectives

This proposal outlines the design and implementation plan for adding comprehensive motor database support with electrical characteristics tracking to mjlab. The system will enable users to:
- Model realistic electrical constraints and thermal behavior
- Validate energy efficiency of control policies

### Key Metrics

- **Implementation**: ~2200 lines of code across 8 new files
- **Testing**: ~1550 lines of test code with 43 unit/integration tests
- **Coverage Target**: >90% code coverage
- **Performance Target**: <10% overhead vs existing DC motor actuator
- **Complexity**: Moderate to high (requires electrical dynamics, numerical integration, thermal modeling)

## Motivation

### Current State

mjlab's actuator system currently provides:
- Basic mechanical properties (reflected inertia, effort limits, gear ratios)
- DC motor torque-speed curves
- No electrical circuit modeling
- No per-step electrical state tracking
- No thermal dynamics

### Why This Feature?

1. **Realistic Power Modeling**: Essential for battery life estimation, power budget analysis, and energy-efficient control
2. **Hardware-in-Loop Preparation**: Enables sim-to-real transfer by modeling actual electrical constraints
3. **Thermal Management**: Critical for long-duration tasks and identifying thermal failure modes
4. **Motor Selection**: Helps users evaluate different motor options during design phase
5. **Dataset Reference**: Provides standardized motor database for reproducible research

### Use Cases

- **Robotics Research**: Train policies that optimize for energy efficiency
- **Hardware Design**: Evaluate motor selection and thermal management strategies
- **Long-Duration Tasks**: Model battery depletion and thermal limits
- **Sim-to-Real Transfer**: Match simulation electrical behavior to real hardware

## Motor Database Search Paths

### Overview

Motor specifications can be stored in multiple locations, allowing flexibility for:
- **Built-in motors**: Shipped with mjlab for common actuators
- **Community motors**: Shared via GitHub repositories
- **Custom motors**: User-specific or project-specific definitions
- **Private motors**: Organization-internal motor databases

### Search Path Priority

When loading a motor by ID, mjlab searches in this order:

1. **Explicit path/URL** (if provided in `load_motor_spec()`)
2. **User directory**: `~/.mjlab/motors/`
3. **Project directory**: `./motors/` (relative to current working directory)
4. **Environment variable**: `$MJLAB_MOTOR_PATH` (colon-separated paths)
5. **Added paths**: Paths added via `add_motor_database_path()`
6. **Built-in database**: `<mjlab_install>/motor_database/motors/`

### Configuration

**Environment Variables**:
```bash
# Add multiple motor database paths
export MJLAB_MOTOR_PATH="/path/to/motors:/another/path/to/motors"

# Point to cloned GitHub repo
export MJLAB_MOTOR_PATH="$HOME/repos/mjlab-motors/community:$MJLAB_MOTOR_PATH"
**Configuration File** (`~/.mjlab/config.yaml`):
```yaml
motor_database:
  search_paths:
    - /path/to/custom/motors
    - ~/projects/robot/motors
  remote_repositories:
      branch: main
      cache_dir: ~/.mjlab/cache/motors/community
```
```python
from mjlab.motor_database import MotorDatabaseConfig

config = MotorDatabaseConfig.get()
config.add_search_path("/path/to/motors")
config.add_remote_repository(
```

### External Repository Structure
```
motor-database-repo/
├── README.md
│   │   ├── unitree_7520_14.json
│   │   ├── unitree_5020_9.json
│   │   └── README.md
│   └── custom/
│       └── my_custom_motor.json
├── assets/
│   ├── unitree/
│   │   ├── 7520-14.step
│   │   └── 7520-14.stl
│   └── maxon/
│       └── ec90_flat_70w.step
└── tests/
    └── test_motor_specs.py
```

**Example: Clone and use community motors**:
```bash
# Clone MuJoCo Menagerie (includes robot models + motor specs)
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie

# Set environment variable to include Menagerie motors
export MJLAB_MOTOR_PATH="$HOME/mujoco_menagerie:$MJLAB_MOTOR_PATH"

# Or add programmatically
python -c "from mjlab.motor_database import add_motor_database_path; \
           add_motor_database_path('~/mujoco_menagerie')"
```

**Usage in Python**:
```python
# Automatically finds motor from Menagerie
motor = load_motor_spec("g1_7520_14")  # From unitree_g1/motors/

# Or from custom repo
motor = load_motor_spec("custom_motor_123", path="~/my-motors")
```

### Loading from Remote URLs

**Direct URL loading** (no local copy needed):
```python
from mjlab.motor_database import load_motor_spec

# Load from GitHub raw URL
motor = load_motor_spec(
    url="https://raw.githubusercontent.com/mjlab-motors/community/main/motors/unitree/unitree_7520_14.json"
)

# Load from S3 or other cloud storage
motor = load_motor_spec(
    url="https://my-bucket.s3.amazonaws.com/motors/custom_motor.json"
)

# Cache is automatically maintained in ~/.mjlab/cache/
```

**With caching and validation**:
```python
from mjlab.motor_database import load_motor_spec, CachePolicy

motor = load_motor_spec(
    url="https://example.com/motor.json",
    cache_policy=CachePolicy.PREFER_CACHE,  # Use cache if available
    cache_ttl=3600,  # Cache valid for 1 hour
    validate_checksum=True  # Verify integrity
)
```

### Database Loader Implementation

**File**: `src/mjlab/motor_database/database.py`

```python
"""Motor database loader with flexible path resolution."""

import json
import os
from pathlib import Path
from typing import Optional
import urllib.request

from mjlab.motor_database.motor_spec import MotorSpecification

# Default search paths
_SEARCH_PATHS: list[Path] = []

def get_default_search_paths() -> list[Path]:
    """Get default motor database search paths."""
    paths = []

    # 1. User directory
    user_dir = Path.home() / ".mjlab" / "motors"
    if user_dir.exists():
        paths.append(user_dir)

    # 2. Current working directory
    cwd_motors = Path.cwd() / "motors"
    if cwd_motors.exists():
        paths.append(cwd_motors)

    # 3. Environment variable
    if "MJLAB_MOTOR_PATH" in os.environ:
        for path_str in os.environ["MJLAB_MOTOR_PATH"].split(":"):
            path = Path(path_str).expanduser()
            if path.exists():
                paths.append(path)

    # 4. Added paths
    paths.extend(_SEARCH_PATHS)

    # 5. Built-in database
    builtin = Path(__file__).parent / "motors"
    if builtin.exists():
        paths.append(builtin)

    return paths

def add_motor_database_path(path: str | Path) -> None:
    """Add a motor database search path."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Motor database path does not exist: {path}")
    _SEARCH_PATHS.append(path)

def load_motor_spec(
    motor_id: Optional[str] = None,
    *,
    path: Optional[str | Path] = None,
    url: Optional[str] = None,
    file: Optional[str | Path] = None,
) -> MotorSpecification:
    """Load motor specification from various sources.

    Args:
        motor_id: Motor ID to search for in database paths
        path: Explicit directory to search in
        url: Direct URL to motor JSON file
        file: Direct path to motor JSON file

    Returns:
        MotorSpecification instance

    Examples:
        >>> # Load from database by ID
        >>> motor = load_motor_spec("unitree_7520_14")

        >>> # Load from explicit path
        >>> motor = load_motor_spec("unitree_7520_14", path="/custom/motors")

        >>> # Load from URL
        >>> motor = load_motor_spec(url="https://example.com/motor.json")

        >>> # Load from file
        >>> motor = load_motor_spec(file="/absolute/path/motor.json")
    """

    # Method 1: Direct file path
    if file is not None:
        return _load_from_file(Path(file))

    # Method 2: Direct URL
    if url is not None:
        return _load_from_url(url)

    # Method 3: Search by motor_id
    if motor_id is not None:
        if path is not None:
            # Search in explicit path only
            return _load_from_path(motor_id, Path(path))
        else:
            # Search in all default paths
            for search_path in get_default_search_paths():
                try:
                    return _load_from_path(motor_id, search_path)
                except FileNotFoundError:
                    continue

            raise FileNotFoundError(
                f"Motor '{motor_id}' not found in any search path. "
                f"Searched: {get_default_search_paths()}"
            )

    raise ValueError("Must provide motor_id, file, or url")

def _load_from_file(file_path: Path) -> MotorSpecification:
    """Load motor spec from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return MotorSpecification(**data)

def _load_from_url(url: str) -> MotorSpecification:
    """Load motor spec from URL with caching."""
    # Check cache first
    cache_dir = Path.home() / ".mjlab" / "cache" / "motors"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use URL hash as cache key
    import hashlib
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"

    # Download if not cached
    if not cache_file.exists():
        with urllib.request.urlopen(url) as response:
            data = response.read()
        cache_file.write_bytes(data)

    return _load_from_file(cache_file)

def _load_from_path(motor_id: str, search_path: Path) -> MotorSpecification:
    """Load motor spec from search path."""
    # Try common patterns
    patterns = [
        search_path / f"{motor_id}.json",
        search_path / motor_id / f"{motor_id}.json",
        search_path / "**" / f"{motor_id}.json",
    ]

    for pattern in patterns:
        if "*" in str(pattern):
            # Glob pattern
            matches = list(search_path.glob(str(pattern.relative_to(search_path))))
            if matches:
                return _load_from_file(matches[0])
        else:
            # Direct path
            if pattern.exists():
                return _load_from_file(pattern)

    raise FileNotFoundError(f"Motor '{motor_id}' not found in {search_path}")
```

### Example: Loading Motors from MuJoCo Menagerie

**Setup**:
```bash
# Clone MuJoCo Menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie

# Add to search path
export MJLAB_MOTOR_PATH="$HOME/mujoco_menagerie:$MJLAB_MOTOR_PATH"
```

**Load robot with motors from Menagerie**:
```python
from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.motor_database import load_motor_spec
from mjlab.scene import SceneCfg

# Motors are discovered from Menagerie automatically
hip_motor = load_motor_spec("g1_7520_14")      # From unitree_g1/motors/
ankle_motor = load_motor_spec("g1_5020_9")    # From unitree_g1/motors/

# Configure robot with Menagerie motors
robot_cfg = EntityCfg(
    spec_fn=lambda: load_robot_spec_from_menagerie("unitree_g1"),
    articulation=EntityArticulationInfoCfg(
        actuators=(
            ElectricalMotorActuatorCfg(
                target_names_expr=(".*_hip_.*",),
                motor_spec=hip_motor,
            ),
            ElectricalMotorActuatorCfg(
                target_names_expr=(".*_ankle_.*",),
                motor_spec=ankle_motor,
            ),
        ),
    ),
)
```

**Contribute back to Menagerie**:
```bash
# After calibrating motors on your robot
cd ~/mujoco_menagerie
git checkout -b add-g1-calibrated-motors

# Add calibrated motor specs
cp my_calibrated_g1_7520_14.json unitree_g1/motors/g1_7520_14_calibrated_serial_001.json

git add unitree_g1/motors/
git commit -m "Add calibrated motor specs for G1 serial 001"
git push origin add-g1-calibrated-motors

# Create PR to google-deepmind/mujoco_menagerie
```

## Design Overview

### Architecture

The system consists of 5 major components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Motor Database                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  JSON Motor Specs (unitree_7520_14.json, ...)         │ │
│  │  - Electrical: R, L, Kt, Ke, voltage range            │ │
│  │  - Mechanical: gear ratio, inertia, torque limits     │ │
│  │  - Thermal: R_th, τ_th, T_max                         │ │
│  │  - 3D Assets: STEP/STL file references                │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              ElectricalMotorActuator                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Per-Step Electrical Dynamics (compute)                │ │
│  │  - Back-EMF: V_bemf = Ke * ω                          │ │
│  │  - RL Circuit: I = (V - V_bemf + L*I_old/dt)/(R+L/dt) │ │
│  │  - Torque: τ = Kt * I                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Thermal Integration (update)                          │ │
│  │  - Power: P = I²*R                                     │ │
│  │  - Temp: dT/dt = (P - (T-T_amb)/R_th) / τ_th          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Electrical Metrics                         │
│  - Total power consumption (W)                              │
│  - Peak motor temperature (°C)                              │
│  - Total energy consumed (J)                                │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Motor Database Infrastructure

**Location**: `src/mjlab/motor_database/`

**Files**:
- `motor_spec.py` - Motor specification dataclass
- `database.py` - Database loader with flexible path resolution
- `motors/` - Built-in JSON motor specifications (ships with mjlab)

**External Motor Repositories**:
Motor specifications can be organized in separate repositories:
- **MuJoCo Menagerie**: `https://github.com/google-deepmind/mujoco_menagerie` - Official robot models with motor specs
- Community motors: `https://github.com/mjlab-motors/community`
- Manufacturer repos: `https://github.com/unitree-robotics/motor-specs`
- Private repos: Your organization's internal motor database
- Local directories: Project-specific motor definitions

**MuJoCo Menagerie Integration**:
The [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) is a collection of high-quality robot models. We propose adding motor specifications alongside each robot model:

```
mujoco_menagerie/
├── unitree_go2/
│   ├── go2.xml
│   ├── scene.xml
│   └── motors/              # NEW: Motor specs for this robot
│       ├── go2_hip_motor.json
│       └── go2_knee_motor.json
├── unitree_g1/
│   ├── g1.xml
│   └── motors/
│       ├── g1_7520_14.json  # Hip motors
│       └── g1_5020_9.json   # Ankle motors
```

This allows users to:
- Load robot models from Menagerie with accurate motor specs
- Contribute calibrated motor parameters back to the community
- Ensure motor specs stay synchronized with robot models

**Motor Specification Schema**:
```python
@dataclass
class MotorSpecification:
    # Identity
    motor_id: str                              # "unitree_7520_14"
    manufacturer: str                          # "Unitree"
    model: str                                 # "7520-14"

    # 3D Assets
    step_file: Optional[str]                   # Path to STEP file
    stl_file: Optional[str]                    # Path to STL file

    # Mechanical Properties
    gear_ratio: float                          # 14.5
    reflected_inertia: float                   # kg⋅m² (after gearbox)
    rotation_angle_range: tuple[float, float]  # (min, max) radians

    # Electrical Properties
    voltage_range: tuple[float, float]         # (min, max) V
    resistance: float                          # Ω (winding resistance)
    inductance: float                          # H (winding inductance)
    motor_constant_kt: float                   # N⋅m/A (torque constant)
    motor_constant_ke: float                   # V⋅s/rad (back-EMF constant)

    # Performance Characteristics
    stall_torque: float                        # N⋅m (at zero speed)
    peak_torque: float                         # N⋅m (instantaneous max)
    continuous_torque: float                   # N⋅m (continuous rating)
    no_load_speed: float                       # rad/s (at zero torque)
    no_load_current: float                     # A
    stall_current: float                       # A
    operating_current: float                   # A (nominal)

    # Thermal Properties
    thermal_resistance: float                  # °C/W (junction to ambient)
    thermal_time_constant: float               # s (first-order model)
    max_winding_temperature: float             # °C
    ambient_temperature: float                 # °C

    # Feedback & Control
    encoder_resolution: int                    # counts/rev
    encoder_type: str                          # "incremental" | "absolute"
    feedback_sensors: list[str]                # ["position", "velocity", "current"]
    protocol: str                              # "PWM" | "CAN" | "UART"
    protocol_params: dict                      # Protocol-specific config
```

**Example JSON** (`motors/unitree_7520_14.json`):
```json
{
  "motor_id": "unitree_7520_14",
  "manufacturer": "Unitree",
  "model": "7520-14",
  "step_file": "data/motor_assets/unitree/7520-14.step",
  "stl_file": "data/motor_assets/unitree/7520-14.stl",
  "gear_ratio": 14.5,
  "reflected_inertia": 0.0015,
  "rotation_angle_range": [-3.14159, 3.14159],
  "voltage_range": [0.0, 24.0],
  "resistance": 0.18,
  "inductance": 0.00015,
  "motor_constant_kt": 0.105,
  "motor_constant_ke": 0.105,
  "stall_torque": 88.0,
  "peak_torque": 120.0,
  "continuous_torque": 88.0,
  "no_load_speed": 32.0,
  "no_load_current": 0.5,
  "stall_current": 15.0,
  "operating_current": 3.0,
  "thermal_resistance": 5.0,
  "thermal_time_constant": 300.0,
  "max_winding_temperature": 120.0,
  "ambient_temperature": 25.0,
  "encoder_resolution": 2048,
  "encoder_type": "incremental",
  "feedback_sensors": ["position", "velocity"],
  "protocol": "CAN",
  "protocol_params": {"baudrate": 1000000}
}
```

**Flexible Loading API**:
```python
from mjlab.motor_database import load_motor_spec, add_motor_database_path

# Method 1: Load from built-in database
motor = load_motor_spec("unitree_7520_14")

# Method 2: Load from external path (relative or absolute)
motor = load_motor_spec("unitree_7520_14", path="/path/to/motor/database")

# Method 3: Load from URL (GitHub, S3, etc.)
motor = load_motor_spec(
    "unitree_7520_14",
    url="https://raw.githubusercontent.com/user/motors/main/unitree_7520_14.json"
)

# Method 4: Add search path permanently
add_motor_database_path("/path/to/custom/motors")
motor = load_motor_spec("custom_motor_123")  # Searches all paths

# Method 5: Load from file directly
motor = load_motor_spec(file="/absolute/path/to/motor.json")

print(f"Continuous torque: {motor.continuous_torque} N⋅m")
print(f"Winding resistance: {motor.resistance} Ω")
```

#### 2. Electrical Motor Actuator

**Location**: `src/mjlab/actuator/electrical_motor_actuator.py`

**Design**: Extends `DcMotorActuator` with full electrical and thermal modeling

**State Tracking** (per-environment, per-joint tensors):
- `current` - Current draw (A)
- `voltage` - Terminal voltage (V)
- `back_emf` - Back-EMF voltage (V)
- `power_dissipation` - Resistive power loss (W)
- `winding_temperature` - Temperature (°C)

**Electrical Dynamics** (in `compute()`):

```python
# 1. Back-EMF from velocity
V_bemf = Ke * ω

# 2. Desired torque from PD controller
τ_desired = Kp * (θ_target - θ) + Kd * (ω_target - ω)

# 3. Required current
I_target = τ_desired / Kt

# 4. RL circuit integration (semi-implicit for stability)
# V = I*R + L*dI/dt + V_bemf
# Rearranged: I_new = (V - V_bemf + L*I_old/dt) / (R + L/dt)
V_terminal = I_target * R + V_bemf + L * (I_target - I_old) / dt

# 5. Voltage clamping (battery limits)
V_terminal = clamp(V_terminal, V_min, V_max)

# 6. Actual current from clamped voltage
I_actual = (V_terminal - V_bemf + L * I_old / dt) / (R + L / dt)

# 7. Actual torque
τ_actual = Kt * I_actual
```

**Thermal Dynamics** (in `update()`):

```python
# Power dissipation
P_loss = I² * R

# First-order thermal RC model
# Heat capacity: C_th = τ_th / R_th
# dT/dt = (P_in - P_out) / C_th
#       = (P_loss - (T - T_amb) / R_th) / (τ_th / R_th)
#       = (P_loss - (T - T_amb) / R_th) * R_th / τ_th

heat_in = P_loss
heat_out = (T - T_amb) / R_th

dT_dt = (heat_in - heat_out) / τ_th
T_new = T_old + dT_dt * dt

# Temperature clamping (safety limit)
T_new = clamp(T_new, T_amb, T_max)
```

**Configuration**:
```python
from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.motor_database import load_motor_spec

actuator_cfg = ElectricalMotorActuatorCfg(
    target_names_expr=(".*_hip_.*", ".*_knee_.*"),
    motor_spec=load_motor_spec("unitree_7520_14"),
)
```

#### 3. XML Integration (Phase 1)

**Location**: `src/mjlab/motor_database/xml_integration.py`

**Purpose**: Enable storing and reading motor_spec references in MuJoCo XML files for sharing via MuJoCo Menagerie.

**Storage Format**: Uses MuJoCo's `<custom><text>` mechanism (official feature, backward compatible):

```xml
<mujoco model="unitree_g1">
  <actuator>
    <motor name="left_hip_motor" joint="left_hip_joint" gear="14.5"/>
    <motor name="right_hip_motor" joint="right_hip_joint" gear="14.5"/>
  </actuator>

  <!-- Motor specifications stored as custom text data -->
  <custom>
    <text name="motor_left_hip_motor" data="motor_spec:unitree_7520_14"/>
    <text name="motor_right_hip_motor" data="motor_spec:unitree_7520_14"/>
  </custom>
</mujoco>
```

**API Functions**:

```python
from mjlab.motor_database import (
    write_motor_spec_to_xml,
    parse_motor_specs_from_xml,
    get_motor_spec,
    has_motor_spec,
)

# Writing motor specs to XML
spec = mujoco.MjSpec.from_file("robot.xml")
write_motor_spec_to_xml(spec, "left_hip_motor", "unitree_7520_14")
xml_output = spec.to_xml()  # Contains motor_spec in <custom><text>

# Reading motor specs from XML
spec = mujoco.MjSpec.from_file("robot_with_motors.xml")
motor_specs = parse_motor_specs_from_xml(spec)
# {'left_hip_motor': 'unitree_7520_14', ...}

# Query individual actuators
if has_motor_spec(spec, "left_hip_motor"):
    motor_id = get_motor_spec(spec, "left_hip_motor")
    motor = load_motor_spec(motor_id)
```

**Benefits**:
- ✅ **Backward compatible**: Standard MuJoCo loads XMLs without errors
- ✅ **Preserves through roundtrips**: XML write/read cycles maintain motor_spec
- ✅ **MuJoCo Menagerie ready**: Share robot XMLs with motor specifications
- ✅ **Auto-loading** (Phase 2): mjlab can auto-detect and create electrical actuators

**Why `<custom><text>` and not attributes?**

MuJoCo strictly validates XML schema and rejects unknown attributes:
```xml
<!-- ❌ This BREAKS - MuJoCo rejects it -->
<motor name="motor1" joint="joint1" motor_spec="unitree_7520_14"/>

<!-- ✅ This WORKS - official MuJoCo feature -->
<custom>
  <text name="motor_motor1" data="motor_spec:unitree_7520_14"/>
</custom>
```

#### 4. MJCF Integration (Phase 2)

**Approach**: Motors are referenced in the actuator configuration (Python-side), not directly in MJCF.

**During `edit_spec()`**:
1. Look up motor spec from database
2. Apply motor parameters to MuJoCo spec (gear ratio, armature, effort limits)
3. Store motor spec reference on actuator instance

**Example**:
```python
def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    motor = self.cfg.motor_spec

    for target_name in target_names:
        actuator = create_motor_actuator(
            spec,
            target_name,
            effort_limit=motor.continuous_torque,
            gear=motor.gear_ratio,
            armature=motor.reflected_inertia,
        )
        self._mjs_actuators.append(actuator)
```

#### 5. Electrical Metrics

**Location**: `src/mjlab/metrics/electrical_metrics.py`

**Metrics**:

```python
class ElectricalPowerTerm(MetricsTerm):
    """Total electrical power consumption across all motors (W)."""

    def compute(self, *, dt: float, **kwargs) -> torch.Tensor:
        power = torch.zeros(self._env.num_envs, device=self._env.device)
        for entity_name in self.cfg.entity_names:
            entity = self._env._scene.entities[entity_name]
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    # P = V * I
                    power += (actuator.voltage * actuator.current).sum(dim=1)
        return power

class MotorTemperatureTerm(MetricsTerm):
    """Maximum winding temperature across all motors (°C)."""

    def compute(self, *, dt: float, **kwargs) -> torch.Tensor:
        max_temp = torch.full(
            (self._env.num_envs,),
            self._env._scene.entities[self.cfg.entity_names[0]]
                ._actuators[0].motor_spec.ambient_temperature,
            device=self._env.device
        )
        for entity_name in self.cfg.entity_names:
            entity = self._env._scene.entities[entity_name]
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    max_temp = torch.maximum(
                        max_temp,
                        actuator.winding_temperature.max(dim=1).values
                    )
        return max_temp

class TotalEnergyTerm(MetricsTerm):
    """Cumulative energy consumption over episode (J)."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._cumulative_energy = torch.zeros(env.num_envs, device=env.device)

    def compute(self, *, dt: float, **kwargs) -> torch.Tensor:
        power = self._compute_power()  # Similar to ElectricalPowerTerm
        self._cumulative_energy += power * dt
        return self._cumulative_energy

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._cumulative_energy.zero_()
        else:
            self._cumulative_energy[env_ids] = 0.0
```

#### 6. Documentation & Examples

**Documentation Updates**:
- Add "Motor Database" section to `docs/source/actuators.rst`
- Include usage examples and electrical model equations
- Document motor specification schema

**Example Script** (`examples/motor_database_example.py`):
```python
"""Example: Using motor database with electrical tracking."""

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from mjlab.managers import MetricsManagerCfg, MetricsTermCfg
from mjlab.metrics.electrical_metrics import (
    ElectricalPowerTerm,
    MotorTemperatureTerm,
)
from mjlab.motor_database import load_motor_spec

# Load motor from database
hip_motor = load_motor_spec("unitree_7520_14")
knee_motor = load_motor_spec("unitree_5020_9")

# Configure robot with electrical actuators
robot_cfg = EntityCfg(
    spec_fn=lambda: load_robot_spec("g1_humanoid"),
    articulation=EntityArticulationInfoCfg(
        actuators=(
            ElectricalMotorActuatorCfg(
                target_names_expr=(".*_hip_.*",),
                motor_spec=hip_motor,
            ),
            ElectricalMotorActuatorCfg(
                target_names_expr=(".*_knee_.*",),
                motor_spec=knee_motor,
            ),
        ),
    ),
)

# Configure metrics to track electrical characteristics
metrics_cfg = MetricsManagerCfg(
    metrics=(
        MetricsTermCfg(
            func=ElectricalPowerTerm,
            name="power",
            entity_names=("robot",),
        ),
        MetricsTermCfg(
            func=MotorTemperatureTerm,
            name="temperature",
            entity_names=("robot",),
        ),
    ),
)

# Create environment
env_cfg = ManagerBasedRLEnvCfg(
    scene=SceneCfg(entities={"robot": robot_cfg}),
    metrics=metrics_cfg,
)
env = env_cfg.build()

# Run simulation
for _ in range(1000):
    obs, _, _, _, info = env.step(actions)

    # Access electrical metrics
    power = info["metrics"]["power"]  # (num_envs,) tensor in W
    temp = info["metrics"]["temperature"]  # (num_envs,) tensor in °C

    # Access actuator electrical state directly
    for actuator in env.scene.entities["robot"]._actuators:
        if isinstance(actuator, ElectricalMotorActuator):
            current = actuator.current  # (num_envs, num_joints)
            voltage = actuator.voltage  # (num_envs, num_joints)
            power_dissipation = actuator.power_dissipation  # (num_envs, num_joints)
```

## Implementation Plan

### Phase 1: Foundation + XML Integration

**Goal**: Establish motor database infrastructure with XML read/write support

**Tasks**:
1. Design and implement `MotorSpecification` dataclass
2. Implement JSON database loader with flexible path resolution
3. **Implement XML integration for motor_spec references**
4. Create 3 example motor specs:
   - Unitree 7520-14 (high-torque)
   - Unitree 5020-9 (mid-range)
   - Test motor (simplified for unit tests)
5. Write motor database tests (~18 tests)
6. Write XML integration tests (~14 tests)

**Deliverables**:
- `src/mjlab/motor_database/motor_spec.py`
- `src/mjlab/motor_database/database.py`
- **`src/mjlab/motor_database/xml_integration.py`** (NEW)
- `src/mjlab/motor_database/motors/*.json` (3 motor specs)
- `tests/test_motor_database.py`
- **`tests/test_motor_xml.py`** (NEW)

**Validation**:
```bash
uv run pytest tests/test_motor_database.py -v
uv run pytest tests/test_motor_xml.py -v
```

**Status**: ✅ **COMPLETED** (32 tests passing)

### Phase 2: Electrical Actuator

**Goal**: Implement electrical actuator with full RL circuit and thermal modeling

**Tasks**:
1. Implement `ElectricalMotorActuator` base structure
2. Add RL circuit electrical dynamics with semi-implicit integration
3. Add thermal dynamics integration
4. Test numerical stability across different timesteps
5. Optimize for GPU/batched operations
6. Write comprehensive tests:
   - RL circuit dynamics tests (~4 tests)
   - Thermal dynamics tests (~5 tests)
   - PD integration tests (~4 tests)
   - Multi-environment batching tests (~2 tests)

**Deliverables**:
- `src/mjlab/actuator/electrical_motor_actuator.py`
- `tests/test_electrical_motor_actuator.py`

**Validation**:
```bash
uv run pytest tests/test_electrical_motor_actuator.py -v
uv run pytest tests/test_electrical_motor_actuator.py::test_current_response_time -v
```

**Status**: ✅ **COMPLETED** (13 tests passing, tagged as phase2-core and phase2-tests)

### Phase 2 (B): Battery System and Power Management

**Goal**: Add comprehensive battery modeling to enable power-limited robotic simulations with realistic voltage sag, state-of-charge tracking, and thermal dynamics.

**Motivation**: While the ElectricalMotorActuator accurately models individual motor electrical dynamics, there is no battery system to:
- Track total power consumption across multiple motors
- Model realistic voltage drop under load (voltage sag from internal resistance)
- Simulate battery discharge and state-of-charge (SOC) evolution
- Limit motor performance when battery voltage drops
- Model inverter efficiency for AC motors (PMSM like in Go2/H1 robots)

**Tasks**:
1. Design and implement `BatterySpecification` dataclass
   - Support LiPo, LiFePO4, Li-ion chemistries
   - Cell configuration (series/parallel)
   - Open-circuit voltage (OCV) curves
   - Internal resistance (SOC and temperature dependent)
   - Thermal properties

2. Implement battery database infrastructure
   - `load_battery_spec()` with flexible path resolution (follows motor database pattern)
   - Search paths: ~/.mjlab/batteries/, ./batteries/, MJLAB_BATTERY_PATH, built-in
   - XML integration for storing battery specs in MuJoCo files
   - Example battery JSON files (LiPo, LiFePO4, test)

3. Implement `BatteryManager` (scene-level power management)
   - Voltage drop physics: V_terminal = V_oc(SOC) - I*R(SOC,T)
   - State-of-charge tracking: dSOC/dt = -I/(Q*3600)
   - Thermal dynamics: dT/dt = (I²R - (T-T_amb)/R_th) / τ_th
   - Current aggregation from all electrical motor actuators
   - Per-environment state tracking (num_envs tensors)

4. Integrate with Scene
   - Add `BatteryManagerCfg` to `SceneCfg`
   - Three-step workflow: compute_voltage() → update motor limits → aggregate_current()
   - Dynamic voltage feedback to motors (battery voltage limits motor performance)

5. Implement inverter support for PMSM motors
   - `InverterCfg` with load-dependent efficiency curves
   - DC-to-AC conversion loss modeling: I_dc = I_ac / efficiency
   - Integrate into `ElectricalMotorActuator`
   - Additional power loss added to thermal budget

6. Write comprehensive tests:
   - Battery database tests (~19 tests): JSON loading, path resolution, XML integration
   - Battery manager tests (~24 tests): voltage drop, SOC, thermal, resistance
   - All existing motor tests continue to pass (~13 tests)

**Deliverables**:
- `src/mjlab/battery_database/battery_spec.py` (163 lines)
- `src/mjlab/battery_database/database.py` (207 lines)
- `src/mjlab/battery_database/xml_integration.py` (120 lines)
- `src/mjlab/battery_database/__init__.py`
- `src/mjlab/battery_database/batteries/*.json` (3 example batteries)
- `src/mjlab/battery/battery_manager.py` (380 lines)
- `src/mjlab/battery/__init__.py`
- `src/mjlab/actuator/inverter.py` (100 lines)
- `src/mjlab/scene/scene.py` (modified for battery integration)
- `src/mjlab/actuator/electrical_motor_actuator.py` (modified for inverter support)
- `tests/test_battery_database.py` (498 lines, 19 tests)
- `tests/test_battery_manager.py` (512 lines, 24 tests)

**Battery Physics Modeled**:

1. **Voltage Drop Model**:
   ```
   V_terminal = V_oc(SOC) - I * R_internal(SOC, T)

   V_oc(SOC) = interpolate(ocv_curve, SOC)
   R_internal(SOC, T) = R_base * f(SOC) * (1 + α·ΔT)
   ```

2. **State of Charge**:
   ```
   dSOC/dt = -I / (Q_capacity * 3600)
   SOC ∈ [min_soc, max_soc]  # Clamped to protect battery
   ```

3. **Thermal Dynamics** (matching motor thermal pattern):
   ```
   P_loss = I² * R_internal
   dT/dt = (P_loss - (T - T_amb) / R_th) / τ_th
   ```

4. **Cell Configuration Physics**:
   ```
   Series (6S):   V_pack = 6 * V_cell,  R_pack = 6 * R_cell
   Parallel (2P): Q_pack = 2 * Q_cell,  I_cell = I_pack / 2
   Combined (6S2P): V = 6*V_cell, Q = 2*Q_cell, R = 6*R_cell/2
   ```

**Inverter Model** (for PMSM motors):
- Load-dependent efficiency curve (50% at no load, 92% at 50% load, 90% at full load)
- DC-side current increased: I_dc = I_ac / η
- Additional power loss: ΔP = P_ac * (1/η - 1)
- Integrated into ElectricalMotorActuator with optional `inverter_cfg`

**Example Usage**:
```python
from mjlab.battery import BatteryManagerCfg
from mjlab.battery_database import load_battery_spec
from mjlab.actuator import ElectricalMotorActuatorCfg, InverterCfg
from mjlab.motor_database import load_motor_spec
from mjlab.scene import SceneCfg

# Load specifications
motor = load_motor_spec("unitree_7520_14")
battery = load_battery_spec("turnigy_6s2p_5000mah")

# Configure inverter for PMSM motors (Go2/H1 style)
inverter = InverterCfg(
    efficiency_curve=[
        (0.0, 0.5),   # 50% at no load
        (0.2, 0.85),  # 85% at 20% load
        (0.5, 0.92),  # 92% at 50% load (peak)
        (0.7, 0.93),  # 93% at 70% load
        (1.0, 0.90),  # 90% at full load
    ]
)

# Configure scene with battery and inverter-driven motors
scene_cfg = SceneCfg(
    num_envs=4,
    entities={
        "robot": EntityCfg(
            articulation=EntityArticulationInfoCfg(
                actuators=(
                    ElectricalMotorActuatorCfg(
                        target_names_expr=(".*_hip_.*",),
                        motor_spec=motor,
                        inverter_cfg=inverter,  # PMSM with DC-to-AC conversion
                    ),
                )
            )
        )
    },
    battery=BatteryManagerCfg(
        battery_spec=battery,
        entity_names=("robot",),
        initial_soc=1.0,
        enable_voltage_feedback=True,
    )
)

# Battery state automatically tracked during simulation:
# - SOC decreases with current draw
# - Voltage sags under load (realistic power constraints)
# - Motor performance limited by battery voltage
# - Temperature rises with I²R losses
# - Inverter losses increase DC-side current
```

**Validation**:
```bash
# Battery database tests (19 tests)
uv run pytest tests/test_battery_database.py -v

# Battery manager physics tests (24 tests)
uv run pytest tests/test_battery_manager.py -v

# All tests together (56 tests: 19 + 24 + 13 motor tests)
uv run pytest tests/test_battery_database.py tests/test_battery_manager.py tests/test_electrical_motor_actuator.py -v

# Type checking and formatting
make check
```

**Key Design Decisions**:

1. **Scene-Level Battery Manager**: Battery is shared across all actuators (realistic), managed at Scene level for aggregation
2. **Dynamic Voltage Limits**: Battery voltage dynamically updates motor voltage limits per-step
3. **OCV Curves**: Non-linear voltage curves for different chemistries (LiPo vs LiFePO4 have different discharge characteristics)
4. **SOC-Dependent Resistance**: R increases 2.5x at empty vs full charge (realistic voltage sag at low SOC)
5. **Inverter Efficiency**: Models DC-to-AC conversion for PMSM motors with load-dependent losses

**Status**: ✅ **COMPLETED** (43 tests passing: 19 database + 24 manager, tagged as phase2-b)

### Phase 3: Automatic Motor/Battery Integration

**Goal**: Make motor and battery physics completely automatic during simulation step, requiring zero manual configuration or physics calculations from users.

**Motivation**:
While Phase 1, 2, and 2(B) provide all necessary building blocks (motor database, electrical actuators, battery system), users still needed to:
- Explicitly configure `ElectricalMotorActuatorCfg` for each motor type
- Manually add `BatteryManagerCfg` to scene
- Understand motor/battery physics for proper integration

This phase eliminates all manual configuration by auto-discovering motor/battery specs from XML and automatically integrating physics updates into the simulation loop.

**Implementation**:

1. **Auto-Discovery in Entity** (`src/mjlab/entity/entity.py`, +50 lines):
   - Added `auto_discover_motors: bool = True` flag to `EntityCfg`
   - Implemented `_auto_discover_motors()` method that:
     - Parses motor specs from XML `<custom><text name="motor_*">` elements
     - Maps MuJoCo actuators to their target joints
     - Groups joints by motor spec ID for efficient batching
     - Creates `ElectricalMotorActuatorCfg` instances automatically
   - Integrated into Entity.__init__ flow (called before manual actuator addition)
   - Only activates if `articulation=None` (manual config takes precedence)

2. **Auto-Discovery in Scene** (`src/mjlab/scene/scene.py`, +30 lines):
   - Added `auto_battery: bool = True` flag to `SceneCfg`
   - Implemented `_auto_discover_battery()` method that:
     - Scans all entity configs for `<custom><text name="battery_*">` elements
     - Parses battery specs before entities are built/attached
     - Creates `BatteryManagerCfg` automatically
     - Links to all entities in scene
   - Explicit `battery` config takes precedence over auto-discovery

3. **Automatic Step Integration**:
   - Verified existing `write_data_to_sim()` handles motor current + battery physics
   - Verified existing `update()` calls battery SOC/thermal updates
   - No new step logic needed - already automatic!

**Test Coverage** (`tests/test_auto_discovery.py`, 7 new tests):
- `test_auto_discover_motors_from_xml` - Verifies motor auto-creation
- `test_auto_discover_multiple_motor_types` - Verifies grouping by motor spec
- `test_auto_discover_motors_disabled` - Verifies flag can disable feature
- `test_auto_discover_battery_from_xml` - Verifies battery auto-creation
- `test_manual_config_takes_precedence` - Verifies precedence system
- `test_no_specs_in_xml` - Verifies graceful handling of missing specs
- `test_battery_manual_config_precedence` - Verifies battery precedence

**Usage Example - Before (Manual):**
```python
from mjlab.scene import Scene, SceneCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.battery import BatteryManagerCfg
from mjlab.motor_database import load_motor_spec
from mjlab.battery_database import load_battery_spec

# 50+ lines of manual configuration...
scene_cfg = SceneCfg(
  num_envs=1,
  entities={
    "robot": EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_file("robot.xml"),
      articulation=EntityArticulationInfoCfg(
        actuators=(
          ElectricalMotorActuatorCfg(
            target_names_expr=(".*hip.*",),
            motor_spec=load_motor_spec("unitree_7520_14"),
            stiffness=200.0,
            damping=10.0,
          ),
          # ... repeat for each motor type ...
        )
      )
    )
  },
  battery=BatteryManagerCfg(
    battery_spec=load_battery_spec("unitree_g1_9ah"),
    entity_names=("robot",),
    initial_soc=1.0,
    enable_voltage_feedback=True,
  )
)
```

**Usage Example - After (Automatic):**
```python
from mjlab.scene import Scene, SceneCfg
from mjlab.entity import EntityCfg
import mujoco

# Auto-discovery handles everything!
scene_cfg = SceneCfg(
  num_envs=1,
  entities={
    "robot": EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_file("robot_with_specs.xml"),
      # auto_discover_motors=True,  # Default
    )
  },
  # auto_battery=True,  # Default
)

scene = Scene(scene_cfg, device="cpu")

# Battery updates happen automatically during:
# scene.write_data_to_sim()  # Handles motor current + battery physics
# sim.step()                  # MuJoCo physics
# scene.update(dt)            # Handles battery SOC/thermal updates
```

**Design Decisions**:

1. **Default to Auto-Discovery**: `auto_discover_motors=True` and `auto_battery=True` by default for maximum convenience
2. **Explicit Config Takes Precedence**: If user provides manual `articulation` or `battery` config, auto-discovery is skipped entirely
3. **Backward Compatible**: All 79 existing tests continue to pass without modification
4. **Efficient Batching**: Motors with same spec ID grouped into single actuator for GPU performance
5. **Scene-Level Battery**: Battery remains at Scene level to aggregate across all entities
6. **No New Step Logic**: Leverages existing hooks in `write_data_to_sim()` and `update()` methods
7. **Actuator Replacement**: When XML has both existing actuators AND motor specs, auto-discovery deletes the existing actuators before creating electrical motor actuators (prevents name collision)

**Validation**:
```bash
# New auto-discovery tests
uv run pytest tests/test_auto_discovery.py -v  # 7 tests

# Backward compatibility (all existing tests still pass)
uv run pytest tests/test_entity.py -v          # Entity tests
uv run pytest tests/test_scene.py -v           # Scene tests
uv run pytest tests/test_battery_manager.py -v # Battery tests
uv run pytest tests/test_electrical_motor_actuator.py -v  # Motor actuator tests

# Full test suite
uv run pytest tests/ -v  # 770 tests total (all passing)
```

**Status**: ✅ **COMPLETED** (770 tests passing: 763 existing + 7 new auto-discovery)

### Phase 4: Validation & Polish

**Goal**: Validate against datasheets and optimize performance

**Tasks**:
1. Performance benchmarking (<10% overhead vs DC motor)
2. Datasheet validation tests (~4 tests)
3. Energy conservation validation
4. Documentation (update actuators.rst)
5. Performance optimization if needed

**Deliverables**:
- `tests/test_electrical_performance.py`
- `tests/validation/test_motor_datasheet_validation.py`
- Updated `docs/source/actuators.rst`

**Validation**:
```bash
uv run pytest tests/test_electrical_performance.py::test_electrical_actuator_overhead -v
uv run pytest tests/validation/test_motor_datasheet_validation.py -v
make check  # Full format/lint/type check
make test   # Full test suite
```

### Phase 5: Final Review

**Goal**: Address issues and finalize

**Tasks**:
1. Fix issues discovered during testing
2. Performance tuning based on benchmarks
3. Documentation polish
4. Final validation

## Testing Strategy

### Test Coverage Target: >90%

**Total Tests**: ~43 unit/integration tests across 6 test files

### Test Categories

#### 1. Motor Database Tests (7 tests)

**File**: `tests/test_motor_database.py`

- Load motor spec from JSON
- Validate required fields
- Handle missing motor ID gracefully
- JSON serialization round-trip
- Optional field defaults
- External file path resolution

#### 2. Electrical Physics Tests (20 tests)

**File**: `tests/test_electrical_motor_actuator.py`

**Basic Electrical Properties**:
- Back-EMF computation (V_bemf = Ke * ω)
- Torque-current relationship (τ = Kt * I)
- Voltage clamping to supply limits
- Current computation from voltage

**RL Circuit Dynamics**:
- Semi-implicit integration stability
- L/R time constant validation
- Steady-state current convergence
- No numerical overshoot

**Thermal Dynamics**:
- Power dissipation (P = I²R)
- Thermal time constant
- Steady-state temperature
- Temperature clamping

**PD Integration**:
- Inherits DC motor behavior
- Voltage limits constrain torque
- Combined effort and voltage limits

**Multi-Environment**:
- Correct tensor batching
- Independent per-env dynamics
- Reset clears state properly

#### 3. Metrics Tests (4 tests)

**File**: `tests/metrics/test_electrical_metrics.py`

- Electrical power term aggregation
- Motor temperature tracking
- Energy accumulation over episode
- Multi-entity metrics

#### 4. Integration Tests (5 tests)

**File**: `tests/integration/test_electrical_simulation.py`

- Full simulation step with electrical tracking
- State persistence across steps
- Energy conservation validation
- Comparison to analytical solutions

#### 5. Performance Tests (3 tests)

**File**: `tests/test_electrical_performance.py`

- Overhead <10% vs DC motor actuator
- Large-scale simulation (1000+ envs)
- GPU memory footprint

#### 6. Datasheet Validation (4 tests)

**File**: `tests/validation/test_motor_datasheet_validation.py`

- Unitree 7520-14 stall torque
- Unitree 7520-14 no-load speed
- Torque-speed curve validation
- Thermal limits

### Test Fixtures

**File**: `tests/fixtures/motors.py`

```python
def create_test_motor_spec(
    resistance: float = 1.0,
    inductance: float = 0.001,
    motor_constant_kt: float = 0.1,
    motor_constant_ke: float = 0.1,
    **kwargs
) -> MotorSpecification:
    """Create motor spec with sensible test defaults."""
    # ... complete defaults for all fields
```

### Continuous Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Test motor database
  run: uv run pytest tests/test_motor_database.py -v

- name: Test electrical actuators
  run: uv run pytest tests/test_electrical_motor_actuator.py -v

- name: Test electrical metrics
  run: uv run pytest tests/metrics/test_electrical_metrics.py -v

- name: Test electrical integration
  run: uv run pytest tests/integration/test_electrical_simulation.py -v
```

## Design Decisions

### 1. Database Format: JSON

**Decision**: Store motor specs as JSON files in `src/mjlab/motor_database/motors/`

**Rationale**:
- Version control friendly (can track changes, create PRs)
- Human-readable and editable
- No external database dependency
- Simple to add new motors
- Can migrate to external DB later if needed

**Alternatives Considered**:
- Python dataclasses in code (not flexible enough)
- YAML (similar to JSON, less Python-native)
- External database (unnecessary complexity)

### 2. Electrical Model: Full RL Circuit

**Decision**: Include inductance in electrical model

**Equation**: `V = I⋅R + L⋅dI/dt + Ke⋅ω`

**Rationale**:
- More physically accurate
- Models transient current response
- Important for high-frequency control
- Semi-implicit integration maintains stability

**Integration Method**: Semi-implicit (backward Euler for L term)
```
I_new = (V - V_bemf + L*I_old/dt) / (R + L/dt)
```

**Alternatives Considered**:
- Resistive only (simpler but less accurate)
- Explicit integration (unstable for large L/R ratios)

### 3. 3D Assets: External References

**Decision**: Motor database stores paths to STEP/STL files

**Rationale**:
- Keeps mjlab package lightweight
- Users can optionally download assets
- Easier to update 3D models independently
- Standard CAD files can be large (10+ MB each)

**Example**:
```json
{
  "motor_id": "unitree_7520_14",
  "step_file": "data/motor_assets/unitree/7520-14.step",
  "stl_file": "data/motor_assets/unitree/7520-14.stl"
}
```

### 4. Wear/Tear: Not Included Initially

**Decision**: Assume constant motor performance (no degradation over time)

**Rationale**:
- Simplifies initial implementation
- Wear models require extensive validation data
- Can be added as optional feature later
- Most RL training doesn't require wear modeling

**Future Work**: Could add optional wear tracking:
- Bearing friction degradation
- Winding resistance increase
- Thermal degradation

### 5. State Storage: On Actuator Instance

**Decision**: Store electrical state as tensors on actuator

**Pattern** (matches existing `IdealPdActuator`):
```python
self.current = torch.zeros((num_envs, num_targets), device=device)
self.voltage = torch.zeros((num_envs, num_targets), device=device)
```

**Rationale**:
- Consistent with existing actuator pattern
- Efficient batched GPU operations
- Easy access for logging and metrics
- Enables per-environment variation

### 6. Thermal Model: First-Order RC

**Decision**: Use simple first-order thermal model

**Equation**: `dT/dt = (P_loss - (T - T_amb)/R_th) / τ_th`

**Rationale**:
- Sufficient accuracy for most applications
- Fast computation
- Parameters easy to obtain from datasheets
- Can be upgraded to multi-node model if needed

**Alternatives Considered**:
- Multi-node thermal network (unnecessary complexity)
- Lookup tables (not generalizable)

## Performance Considerations

### Target: <10% Overhead vs DC Motor Actuator

**Optimization Strategies**:

1. **Batched Operations**: All computations vectorized across environments
2. **In-Place Updates**: Minimize memory allocations
3. **GPU-Friendly**: Pure PyTorch operations, no Python loops
4. **Cached Constants**: Pre-compute motor constants during initialization

**Benchmark Test**:
```python
def test_electrical_actuator_overhead():
    # Run 1000 steps with 100 environments
    # Compare DC motor vs electrical motor timing
    assert (elec_time - dc_time) / dc_time < 0.10
```

### Memory Footprint

**Per actuator per environment**:
- 5 electrical state tensors × (num_envs × num_joints) × 4 bytes
- Example: 1000 envs, 12 joints → 240 KB

**Acceptable for modern GPUs**: Even 10,000 environments → 2.4 MB per actuator

## Calibration

### Overview

Real motors often deviate from datasheet specifications due to manufacturing tolerances, wear, or environmental conditions. To enable accurate sim-to-real transfer, mjlab supports **motor parameter calibration** through system identification.

### Workflow

```
1. Initial Simulation (mjlab)
   ↓ Collect: joint positions, velocities, commanded torques, predicted currents

2. Real Robot Execution
   ↓ Measure: actual current draw, voltage, (optional) torque feedback

3. Parameter Optimization
   ↓ Use ML/optimization to fit motor parameters

4. Export Calibrated Motor Spec
   ↓ Generate new JSON variant (e.g., "unitree_7520_14_calibrated.json")
```

### Data Collection

**Simulation Side** (mjlab):
```python
# Run trajectory in simulation
trajectory_data = []
for step in trajectory:
    env.step(action)

    # Collect electrical predictions
    for actuator in env.scene.entities["robot"]._actuators:
        if isinstance(actuator, ElectricalMotorActuator):
            trajectory_data.append({
                'time': step * dt,
                'joint_pos': actuator.entity._data.joint_pos.cpu(),
                'joint_vel': actuator.entity._data.joint_vel.cpu(),
                'commanded_torque': sim.data.ctrl.cpu(),
                'predicted_current': actuator.current.cpu(),
                'predicted_voltage': actuator.voltage.cpu(),
                'predicted_power': actuator.power_dissipation.cpu(),
            })

# Save for comparison
np.save('sim_trajectory.npy', trajectory_data)
```

**Real Robot Side**:
```python
# Execute same trajectory on real robot
real_data = []
for step in trajectory:
    robot.send_command(action)

    # Measure actual electrical characteristics
    real_data.append({
        'time': step * dt,
        'joint_pos': robot.get_joint_positions(),
        'joint_vel': robot.get_joint_velocities(),
        'measured_current': robot.get_motor_currents(),      # From motor drivers
        'measured_voltage': robot.get_motor_voltages(),      # If available
        'measured_torque': robot.get_joint_torques(),        # If available (force/torque sensors)
    })

# Save for parameter estimation
np.save('real_trajectory.npy', real_data)
```

### Parameter Estimation Methods

#### Method 1: Least Squares Optimization

**Optimize electrical parameters** to minimize prediction error:

```python
from scipy.optimize import minimize

def objective(params, sim_data, real_data):
    """Compute error between predicted and measured currents."""
    R, L, Kt, Ke = params

    error = 0.0
    for sim, real in zip(sim_data, real_data):
        # Re-compute predicted current with new parameters
        V_bemf = Ke * sim['joint_vel']
        I_predicted = (sim['commanded_torque'] / Kt)
        # ... (simplified, actual would include RL dynamics)

        # Compare to measured
        error += np.sum((I_predicted - real['measured_current'])**2)

    return error

# Initial guess from datasheet
initial_params = [R_datasheet, L_datasheet, Kt_datasheet, Ke_datasheet]

# Optimize
result = minimize(objective, initial_params, args=(sim_data, real_data))
R_cal, L_cal, Kt_cal, Ke_cal = result.x
```

#### Method 2: Neural Network Calibration

**Learn a correction model** for motor parameters:

```python
import torch
import torch.nn as nn

class MotorCalibrationNet(nn.Module):
    """Predicts parameter corrections based on operating conditions."""

    def __init__(self):
        super().__init__()
        # Input: joint_pos, joint_vel, temperature, battery_voltage
        # Output: corrections to R, L, Kt, Ke
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # δR, δL, δKt, δKe
        )

    def forward(self, state):
        return self.net(state)

# Train on sim-to-real error
model = MotorCalibrationNet()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    for sim, real in zip(sim_data, real_data):
        # Predict parameter corrections
        state = torch.tensor([sim['joint_pos'], sim['joint_vel'],
                             sim['temperature'], sim['voltage']])
        corrections = model(state)

        # Apply corrections and compute current
        R_adj = R_datasheet + corrections[0]
        L_adj = L_datasheet + corrections[1]
        Kt_adj = Kt_datasheet + corrections[2]
        Ke_adj = Ke_datasheet + corrections[3]

        # Compute loss against measured current
        I_predicted = compute_current(sim, R_adj, L_adj, Kt_adj, Ke_adj)
        loss = (I_predicted - real['measured_current'])**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### Method 3: Bayesian Parameter Estimation

**Uncertainty quantification** for parameters:

```python
import pyro
import pyro.distributions as dist

def model(sim_data, real_data):
    # Priors from datasheet ± tolerance
    R = pyro.sample('R', dist.Normal(R_datasheet, R_tolerance))
    L = pyro.sample('L', dist.Normal(L_datasheet, L_tolerance))
    Kt = pyro.sample('Kt', dist.Normal(Kt_datasheet, Kt_tolerance))
    Ke = pyro.sample('Ke', dist.Normal(Ke_datasheet, Ke_tolerance))

    # Likelihood
    for sim, real in zip(sim_data, real_data):
        I_predicted = compute_current(sim, R, L, Kt, Ke)
        pyro.sample('obs', dist.Normal(I_predicted, sigma_noise),
                   obs=real['measured_current'])

# Run MCMC inference
from pyro.infer import MCMC, NUTS
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000)
mcmc.run(sim_data, real_data)

# Get posterior distributions
samples = mcmc.get_samples()
R_posterior = samples['R']  # Distribution, not point estimate
```

### Excitation Trajectories

**Design trajectories** to excite all motor dynamics:

```python
def generate_identification_trajectory():
    """Generate trajectory that excites motor dynamics across operating range."""

    trajectory = []

    # 1. Velocity sweep (identify back-EMF constant Ke)
    for vel in np.linspace(0, max_vel, 20):
        trajectory.append(constant_velocity_motion(vel, duration=1.0))

    # 2. Torque sweep (identify torque constant Kt, resistance R)
    for torque in np.linspace(0, max_torque, 20):
        trajectory.append(constant_torque_motion(torque, duration=1.0))

    # 3. Step response (identify inductance L, thermal time constant)
    trajectory.append(step_input(amplitude=max_torque * 0.5))

    # 4. Frequency sweep (identify electrical and thermal dynamics)
    for freq in np.logspace(-1, 2, 30):  # 0.1 Hz to 100 Hz
        trajectory.append(sinusoidal_motion(freq, duration=5.0))

    # 5. Thermal cycling (identify thermal resistance R_th, time constant τ_th)
    trajectory.append(high_power_motion(duration=300.0))  # 5 min heating
    trajectory.append(rest(duration=600.0))  # 10 min cooling

    return trajectory
```

### Validation

**Validate calibrated parameters** on held-out trajectories:

```python
def validate_calibration(calibrated_motor_spec, validation_trajectories):
    """Check if calibrated parameters improve prediction accuracy."""

    # Run validation trajectories in sim with calibrated parameters
    env = create_env_with_calibrated_motor(calibrated_motor_spec)

    errors_before = []
    errors_after = []

    for traj in validation_trajectories:
        # Simulate with original parameters
        sim_original = run_sim(env_original, traj)

        # Simulate with calibrated parameters
        sim_calibrated = run_sim(env, traj)

        # Run on real robot
        real = run_real(robot, traj)

        # Compare prediction errors
        error_before = np.mean((sim_original['current'] - real['current'])**2)
        error_after = np.mean((sim_calibrated['current'] - real['current'])**2)

        errors_before.append(error_before)
        errors_after.append(error_after)

    improvement = (np.mean(errors_before) - np.mean(errors_after)) / np.mean(errors_before)
    print(f"Prediction error reduced by {improvement*100:.1f}%")
```

### Export Calibrated Motor Spec

**Generate new JSON** with calibrated parameters:

```python
def export_calibrated_motor(original_motor_id, calibrated_params, metadata):
    """Create calibrated motor variant."""

    # Load original motor spec
    original = load_motor_spec(original_motor_id)

    # Create calibrated variant
    calibrated = MotorSpecification(
        motor_id=f"{original_motor_id}_calibrated_{metadata['robot_serial']}",
        manufacturer=original.manufacturer,
        model=f"{original.model} (Calibrated)",

        # Update electrical parameters with calibrated values
        resistance=calibrated_params['R'],
        inductance=calibrated_params['L'],
        motor_constant_kt=calibrated_params['Kt'],
        motor_constant_ke=calibrated_params['Ke'],
        thermal_resistance=calibrated_params.get('R_th', original.thermal_resistance),
        thermal_time_constant=calibrated_params.get('tau_th', original.thermal_time_constant),

        # Keep mechanical parameters from original
        gear_ratio=original.gear_ratio,
        reflected_inertia=original.reflected_inertia,
        # ... (other fields)

        # Add calibration metadata
        calibration_metadata={
            'original_motor_id': original_motor_id,
            'calibration_date': metadata['date'],
            'robot_serial': metadata['robot_serial'],
            'calibration_method': metadata['method'],  # 'least_squares', 'neural_net', 'bayesian'
            'validation_error': metadata['validation_error'],
            'trajectories_used': metadata['trajectories'],
        }
    )

    # Save to JSON
    output_path = f"motors/{calibrated.motor_id}.json"
    with open(output_path, 'w') as f:
        json.dump(dataclasses.asdict(calibrated), f, indent=2)

    print(f"Calibrated motor saved: {output_path}")
    return calibrated
```

### Integration with mjlab

**Proposed API** for system identification:

```python
from mjlab.motor_database import MotorCalibrator

# 1. Collect simulation data
calibrator = MotorCalibrator(
    motor_id="unitree_7520_14",
    robot_cfg=robot_cfg,
)

sim_data = calibrator.collect_simulation_data(
    trajectories=identification_trajectories,
    num_envs=1,
)

# 2. Load real robot data
real_data = calibrator.load_real_data("real_trajectory.npy")

# 3. Calibrate parameters
calibrated_params = calibrator.calibrate(
    sim_data=sim_data,
    real_data=real_data,
    method='least_squares',  # or 'neural_net', 'bayesian'
    parameters_to_fit=['resistance', 'inductance', 'motor_constant_kt', 'motor_constant_ke'],
)

# 4. Validate
validation_error = calibrator.validate(
    calibrated_params=calibrated_params,
    validation_trajectories=validation_trajectories,
)

# 5. Export calibrated motor
calibrated_motor = calibrator.export(
    calibrated_params=calibrated_params,
    metadata={
        'robot_serial': 'G1-001',
        'date': '2026-03-20',
        'validation_error': validation_error,
    }
)
```

### Benefits

1. **Accurate Sim-to-Real**: Match simulation electrical behavior to real hardware
2. **Individual Calibration**: Account for manufacturing tolerances and wear
3. **Continuous Improvement**: Re-calibrate as motors age
4. **Community Database**: Share calibrated motor specs for specific robot models
5. **Validation Tool**: Quantify sim-to-real gap in electrical characteristics

### Future Work

- **Online Calibration**: Update parameters during operation
- **Multi-Robot Calibration**: Aggregate data from fleet of robots
- **Temperature Compensation**: Calibrate thermal model parameters
- **Automated Tools**: GUI for data collection and calibration workflow
- **Cloud Database**: Community-contributed calibrated motor specs

## Visualization

### Overview

mjlab includes built-in visualization capabilities through the **Viser** web-based viewer and **metrics plotting** system. The electrical motor system should integrate with these existing tools to display real-time electrical characteristics during simulation.

### Existing Visualization Infrastructure

mjlab provides:
- **ViserPlayViewer**: Web-based 3D viewer with interactive controls
- **ViserTermPlotter**: Real-time plotting of rewards/metrics terms
- **Native MuJoCo Viewer**: Alternative native OpenGL viewer
- **OffscreenRenderer**: For rendering without display

**Key files**:
- `src/mjlab/viewer/viser/viewer.py` - Main Viser viewer
- `src/mjlab/viewer/viser/overlays.py` - Overlay management (rewards, metrics, cameras)
- `src/mjlab/viewer/viser/term_plotter.py` - Real-time term plotting

### Electrical Characteristics Display

#### 1. Real-Time Electrical Metrics Plot

**Integration with existing term plotter**:

```python
# Electrical metrics are already accessible via metrics_manager
# Just need to add them to the viewer

from mjlab.viewer import ViserPlayViewer

viewer = ViserPlayViewer(env, policy)

# Electrical metrics automatically appear in "Metrics" tab:
# - power: Total electrical power (W)
# - temperature: Max motor temperature (°C)
# - energy: Cumulative energy (J)
```

The existing `ViserTermPlotter` will automatically plot electrical metrics alongside other metrics:

```python
# In ViserTermOverlays.setup_tabs() - already exists
if hasattr(self.env.unwrapped, "metrics_manager"):
    term_names = [
        name for name, _ in
        self.env.unwrapped.metrics_manager.get_active_iterable_terms(env_idx)
    ]
    # This will include "power", "temperature", "energy" etc.
    self.metrics_plotter = ViserTermPlotter(
        self.server, term_names, name="Metric", env_idx=env_idx
    )
```

#### 2. Per-Actuator Electrical Overlay

**New overlay for detailed actuator-level visualization**:

**File**: `src/mjlab/viewer/viser/electrical_overlays.py` (NEW)

```python
"""Electrical characteristics overlay for Viser viewer."""

from dataclasses import dataclass
import viser
from mjlab.actuator import ElectricalMotorActuator

@dataclass
class ViserElectricalOverlays:
    """Manage electrical characteristics visualization in Viser viewer."""

    server: viser.ViserServer
    env: Any
    scene: Any

    # GUI elements
    current_folder: Any = None
    voltage_folder: Any = None
    power_folder: Any = None
    temperature_folder: Any = None

    def setup_tabs(self, tabs: Any) -> None:
        """Create electrical tab with per-actuator displays."""

        with tabs.add_tab("Electrical", icon=viser.Icon.BOLT):
            # Summary statistics
            with self.server.gui.add_folder("Summary"):
                self.total_power_display = self.server.gui.add_html("")
                self.max_temp_display = self.server.gui.add_html("")
                self.total_energy_display = self.server.gui.add_html("")

            # Per-actuator details
            with self.server.gui.add_folder("Actuators"):
                self._setup_actuator_displays()

    def _setup_actuator_displays(self) -> None:
        """Create display elements for each electrical actuator."""

        env_idx = self.scene.env_idx
        entities = self.env.unwrapped.scene.entities

        for entity_name, entity in entities.items():
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    # Create folder for this actuator
                    with self.server.gui.add_folder(f"{entity_name} - {actuator.motor_spec.motor_id}"):
                        # Current per joint
                        self.server.gui.add_html(
                            "<b>Current (A)</b>",
                            id=f"{entity_name}_{actuator.motor_spec.motor_id}_current"
                        )

                        # Voltage per joint
                        self.server.gui.add_html(
                            "<b>Voltage (V)</b>",
                            id=f"{entity_name}_{actuator.motor_spec.motor_id}_voltage"
                        )

                        # Temperature per joint
                        self.server.gui.add_html(
                            "<b>Temperature (°C)</b>",
                            id=f"{entity_name}_{actuator.motor_spec.motor_id}_temp"
                        )

                        # Power per joint
                        self.server.gui.add_html(
                            "<b>Power (W)</b>",
                            id=f"{entity_name}_{actuator.motor_spec.motor_id}_power"
                        )

    def update(self) -> None:
        """Update electrical displays with current values."""

        env_idx = self.scene.env_idx
        total_power = 0.0
        max_temp = 0.0

        entities = self.env.unwrapped.scene.entities

        for entity_name, entity in entities.items():
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    # Get values for current environment
                    current = actuator.current[env_idx].cpu().numpy()
                    voltage = actuator.voltage[env_idx].cpu().numpy()
                    temp = actuator.winding_temperature[env_idx].cpu().numpy()
                    power = actuator.power_dissipation[env_idx].cpu().numpy()

                    # Update per-actuator displays
                    motor_id = actuator.motor_spec.motor_id

                    # Format as HTML table
                    self.server.gui.update_html(
                        f"{entity_name}_{motor_id}_current",
                        self._format_joint_values("Current", current, "A")
                    )
                    self.server.gui.update_html(
                        f"{entity_name}_{motor_id}_voltage",
                        self._format_joint_values("Voltage", voltage, "V")
                    )
                    self.server.gui.update_html(
                        f"{entity_name}_{motor_id}_temp",
                        self._format_joint_values("Temp", temp, "°C", warning_threshold=80.0)
                    )
                    self.server.gui.update_html(
                        f"{entity_name}_{motor_id}_power",
                        self._format_joint_values("Power", power, "W")
                    )

                    # Accumulate summary stats
                    total_power += power.sum()
                    max_temp = max(max_temp, temp.max())

        # Update summary
        self.total_power_display.content = f"<b>Total Power:</b> {total_power:.1f} W"
        self.max_temp_display.content = f"<b>Max Temperature:</b> {max_temp:.1f} °C"

    def _format_joint_values(
        self,
        label: str,
        values: np.ndarray,
        unit: str,
        warning_threshold: float | None = None
    ) -> str:
        """Format joint values as HTML with optional color coding."""

        html = "<table style='font-size: 12px;'>"

        for i, val in enumerate(values):
            color = "black"
            if warning_threshold is not None and val > warning_threshold:
                color = "red"

            html += f"<tr><td>Joint {i}:</td><td style='color:{color};'>{val:.2f} {unit}</td></tr>"

        html += "</table>"
        return html
```

#### 3. Visual Indicators in 3D Viewport

**Color-coded motor visualization**:

```python
class ViserElectricalOverlays:

    def update_3d_indicators(self) -> None:
        """Add color-coded visual indicators to motors in 3D view."""

        env_idx = self.scene.env_idx

        for entity_name, entity in self.env.unwrapped.scene.entities.items():
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    temp = actuator.winding_temperature[env_idx].cpu().numpy()

                    for joint_idx, joint_temp in enumerate(temp):
                        # Get joint position in world frame
                        joint_name = actuator.target_names[joint_idx]

                        # Color code by temperature
                        if joint_temp < 50:
                            color = (0, 255, 0)  # Green (cool)
                        elif joint_temp < 80:
                            color = (255, 255, 0)  # Yellow (warm)
                        else:
                            color = (255, 0, 0)  # Red (hot)

                        # Add or update sphere at joint location
                        self.server.scene.add_sphere(
                            name=f"temp_indicator_{entity_name}_{joint_name}",
                            radius=0.02,
                            position=joint_position,
                            color=color,
                            opacity=0.5
                        )
```

#### 4. Electrical Time-Series Dashboard

**Alternative: Standalone electrical monitoring dashboard**:

```python
class ElectricalDashboard:
    """Real-time electrical characteristics dashboard using Plotly."""

    def __init__(self, env, update_rate_hz=10):
        self.env = env
        self.update_rate = update_rate_hz

        # Create Plotly dashboard
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Current Draw (A)",
                "Voltage (V)",
                "Power Dissipation (W)",
                "Temperature (°C)"
            )
        )

        # Initialize traces for each actuator
        self._setup_traces()

        # Start update loop
        self._start_update_loop()

    def _setup_traces(self):
        """Add traces for each electrical actuator."""

        for entity_name, entity in self.env.scene.entities.items():
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    motor_id = actuator.motor_spec.motor_id

                    for joint_idx in range(len(actuator.target_names)):
                        label = f"{entity_name}/{motor_id}/joint_{joint_idx}"

                        # Current
                        self.fig.add_trace(
                            go.Scatter(y=[], name=label, mode='lines'),
                            row=1, col=1
                        )

                        # Voltage
                        self.fig.add_trace(
                            go.Scatter(y=[], name=label, mode='lines'),
                            row=1, col=2
                        )

                        # Power
                        self.fig.add_trace(
                            go.Scatter(y=[], name=label, mode='lines'),
                            row=2, col=1
                        )

                        # Temperature
                        self.fig.add_trace(
                            go.Scatter(y=[], name=label, mode='lines'),
                            row=2, col=2
                        )

    def update(self):
        """Update dashboard with latest electrical data."""

        for entity_name, entity in self.env.scene.entities.items():
            for actuator in entity._actuators:
                if isinstance(actuator, ElectricalMotorActuator):
                    # Append new data to traces
                    current = actuator.current[0].cpu().numpy()
                    voltage = actuator.voltage[0].cpu().numpy()
                    power = actuator.power_dissipation[0].cpu().numpy()
                    temp = actuator.winding_temperature[0].cpu().numpy()

                    # Update Plotly traces...
```

### Integration with ViserPlayViewer

**Add electrical overlays to existing viewer**:

```python
# Modified ViserPlayViewer.__init__()
def __init__(self, env, policy, **kwargs):
    super().__init__(env, policy, **kwargs)

    # Existing overlays
    self._term_overlays = None
    self._camera_overlays = None
    self._debug_overlays = None
    self._contact_overlays = None

    # NEW: Electrical overlays
    self._electrical_overlays: ViserElectricalOverlays | None = None

# Modified ViserPlayViewer.setup()
def setup(self):
    # ... existing setup code ...

    # Setup electrical overlays if any electrical actuators exist
    if self._has_electrical_actuators():
        self._electrical_overlays = ViserElectricalOverlays(
            server=self._server,
            env=self.env,
            scene=self._scene
        )
        self._electrical_overlays.setup_tabs(tabs)

# Modified ViserPlayViewer.step()
def step(self):
    # ... existing step code ...

    # Update electrical displays
    if self._electrical_overlays:
        self._electrical_overlays.update()
```

### Example Usage

```python
from mjlab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from mjlab.viewer import ViserPlayViewer
from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.motor_database import load_motor_spec

# Create environment with electrical actuators
env_cfg = ManagerBasedRLEnvCfg(
    scene=SceneCfg(
        entities={
            "robot": EntityCfg(
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        ElectricalMotorActuatorCfg(
                            target_names_expr=(".*_hip_.*",),
                            motor_spec=load_motor_spec("unitree_7520_14"),
                        ),
                    )
                )
            )
        }
    ),
    metrics=MetricsManagerCfg(
        metrics=(
            MetricsTermCfg(func=ElectricalPowerTerm, name="power"),
            MetricsTermCfg(func=MotorTemperatureTerm, name="temperature"),
        )
    )
)

env = env_cfg.build()

# Launch viewer - electrical tab will appear automatically
viewer = ViserPlayViewer(env, policy)
viewer.run()

# Open browser to http://localhost:8080
# Navigate to "Electrical" tab to see:
# - Total power consumption
# - Max motor temperature
# - Per-actuator current, voltage, power, temperature
# - 3D color-coded temperature indicators

# Navigate to "Metrics" tab to see time-series plots of:
# - Total power (W)
# - Max temperature (°C)
```

### Display Features

**Electrical Tab Contents**:
- ✅ **Summary Panel**: Total power, max temperature, total energy
- ✅ **Per-Actuator Folders**: Current, voltage, power, temperature for each joint
- ✅ **Color Coding**: Red warnings for overheating
- ✅ **Real-Time Updates**: 60 Hz refresh rate

**Metrics Tab Contents** (using existing ViserTermPlotter):
- ✅ **Time-Series Plots**: Power consumption, temperature over time
- ✅ **Multiple Environments**: Switch between parallel environments
- ✅ **History Window**: Configurable plot history length

**3D Viewport Indicators**:
- ✅ **Temperature Color Coding**: Green (cool) → Yellow (warm) → Red (hot)
- ✅ **Joint Markers**: Visual indicators at each actuated joint
- ✅ **Optional**: Current flow arrows, power dissipation particles

### Implementation Priority

**Phase 1 (Core Implementation)**:
- ✅ Electrical metrics integration with existing `ViserTermPlotter`
- ✅ Metrics automatically plotted (no additional code needed)

**Phase 2 (Enhanced Visualization)**:
- Add `ViserElectricalOverlays` for detailed per-actuator display
- Integrate with `ViserPlayViewer` tabs

**Phase 3 (Advanced Features)**:
- 3D visual indicators (color-coded temperature spheres)
- Optional standalone Plotly dashboard
- Export electrical traces to CSV/HDF5

### Future Enhancements

- **Thermal Camera View**: Heatmap overlay showing motor temperatures
- **Power Flow Diagram**: Sankey diagram of energy distribution
- **Historical Analysis**: Compare electrical behavior across episodes
- **Alert System**: Visual/audio warnings for overcurrent or overtemperature
- **Battery Simulation**: Display remaining battery capacity

## Open Questions & Future Work

### Short-Term (Deferred)

1. **Communication Delays**: Should integrate with existing `DelayedActuator`
2. **PWM Resolution**: Could model quantization effects
3. **Thermal Coupling**: Multi-motor thermal interaction

### Long-Term

1. **Backlash Modeling**: Model gear backlash and its effect on electrical transients
   - During backlash: motor spins with minimal load → low current
   - After engagement: full load → current spike
   - Important for cheap gearboxes and direction changes
   - Requires state tracking and torque transmission model
2. **Wear Modeling**: Track cumulative stress and degradation
3. **Multi-Node Thermal**: More detailed thermal networks
4. **Motor Catalog Expansion**: Community-contributed motor database repositories
   - **MuJoCo Menagerie Integration**: Add motor specs to existing robot models in `google-deepmind/mujoco_menagerie`
   - `mjlab-motors/community` - Community motors
   - `mjlab-motors/manufacturers` - Official manufacturer specs
   - Per-manufacturer repos (e.g., `unitree-robotics/motor-specs`)
5. **CAD Import Tools**: Auto-generate motor specs from STEP files
6. **Power Electronics**: Model inverter dynamics and switching losses
7. **Remote Database Sync**: Automatic updates from remote repositories
8. **Motor Discovery**: CLI tool to list available motors across all search paths

## Success Metrics

### Quantitative

- ✅ >90% code coverage on new components
- ✅ <10% performance overhead vs DC motor actuator
- ✅ All tests passing (43 unit/integration tests)
- ✅ Type checking passes (mypy/pyright)
- ✅ Datasheet validation: electrical characteristics match spec within 5%

### Qualitative

- ✅ Clear documentation with usage examples
- ✅ Easy to add new motors to database
- ✅ Intuitive API consistent with existing actuators
- ✅ Enables realistic power consumption analysis

## References

### Motor Datasheets

- Unitree Robotics Motor Specifications
- MuJoCo Actuator Documentation
- DC Motor Theory (Electrical Engineering textbooks)

### Related Work

- Isaac Lab actuator system
- Drake's electrical motor models
- Gazebo's motor plugins
- **MuJoCo Menagerie** - [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) - High-quality robot model collection (ideal location for motor specs)

### MuJoCo Documentation

- [MuJoCo Actuators](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator)
- [MjSpec API](https://mujoco.readthedocs.io/en/stable/APIreference.html#mjspec)

## Appendix: File Structure

```
mjlab/
├── src/mjlab/
│   ├── actuator/
│   │   ├── electrical_motor_actuator.py      # NEW: Electrical actuator
│   │   └── ... (existing files)
│   ├── motor_database/                        # NEW: Motor database
│   │   ├── __init__.py
│   │   ├── motor_spec.py                      # Motor specification
│   │   ├── database.py                        # Database loader
│   │   └── motors/                            # Motor JSON files
│   │       ├── unitree_7520_14.json
│   │       ├── unitree_5020_9.json
│   │       └── test_motor.json
│   └── metrics/
│       ├── electrical_metrics.py              # NEW: Electrical metrics
│       └── ... (existing files)
├── tests/
│   ├── test_motor_database.py                 # NEW: Database tests
│   ├── test_electrical_motor_actuator.py      # NEW: Actuator tests
│   ├── metrics/
│   │   └── test_electrical_metrics.py         # NEW: Metrics tests
│   ├── integration/
│   │   └── test_electrical_simulation.py      # NEW: Integration tests
│   ├── test_electrical_performance.py         # NEW: Performance tests
│   ├── validation/
│   │   └── test_motor_datasheet_validation.py # NEW: Validation tests
│   └── fixtures/
│       └── motors.py                          # NEW: Test fixtures
├── examples/
│   └── motor_database_example.py              # NEW: Complete example
├── docs/
│   ├── source/
│   │   └── actuators.rst                      # MODIFIED: Add motor DB section
│   └── design/
│       └── motor_database_proposal.md         # THIS DOCUMENT
└── data/
    └── motor_assets/                          # NEW: STEP/STL files (external)
        └── unitree/
            ├── 7520-14.step
            └── 7520-14.stl
```

## Appendix: Example Motor Specifications

### Unitree 7520-14 (High Torque)

```json
{
  "motor_id": "unitree_7520_14",
  "manufacturer": "Unitree",
  "model": "7520-14",
  "gear_ratio": 14.5,
  "reflected_inertia": 0.0015,
  "resistance": 0.18,
  "inductance": 0.00015,
  "motor_constant_kt": 0.105,
  "motor_constant_ke": 0.105,
  "stall_torque": 88.0,
  "continuous_torque": 88.0,
  "no_load_speed": 32.0,
  "voltage_range": [0.0, 24.0],
  "thermal_resistance": 5.0,
  "thermal_time_constant": 300.0,
  "max_winding_temperature": 120.0
}
```

### Unitree 5020-9 (Mid-Range)

```json
{
  "motor_id": "unitree_5020_9",
  "manufacturer": "Unitree",
  "model": "5020-9",
  "gear_ratio": 9.0,
  "reflected_inertia": 0.0008,
  "resistance": 0.30,
  "inductance": 0.0001,
  "motor_constant_kt": 0.08,
  "motor_constant_ke": 0.08,
  "stall_torque": 25.0,
  "continuous_torque": 25.0,
  "no_load_speed": 40.0,
  "voltage_range": [0.0, 24.0],
  "thermal_resistance": 8.0,
  "thermal_time_constant": 200.0,
  "max_winding_temperature": 100.0
}
```

---

**End of Proposal**
