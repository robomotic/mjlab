# Motor Database: XML-Driven Design

**Date**: 2026-03-23
**Status**: Design Proposal for Phase 2

## Goal

Enable researchers to share robot XMLs with motor specifications that automatically:
1. Load motor specs from motor database
2. Create electrical actuators
3. Run power/current simulations

## XML-Driven Flow

### 1. XML with Motor Spec References

```xml
<mujoco model="unitree_g1">
  <actuator>
    <!-- Electrical motor actuator - auto-detected by motor_spec attribute -->
    <motor name="left_hip_actuator"
           joint="left_hip_joint"
           gear="14.5"
           motor_spec="unitree_7520_14"/>

    <!-- Traditional actuator - no motor_spec -->
    <motor name="torso_actuator"
           joint="torso_joint"
           gear="1.0"/>
  </actuator>
</mujoco>
```

### 2. Auto-Loading in Python

```python
from mjlab.entity import EntityCfg

# Simple - motor specs detected and loaded automatically
robot = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file("unitree_g1.xml")
)

# mjlab automatically:
# 1. Parses XML and finds motor_spec="unitree_7520_14"
# 2. Loads motor spec: load_motor_spec("unitree_7520_14")
# 3. Creates ElectricalMotorActuator with loaded spec
# 4. Electrical simulation runs automatically
```

### 3. Manual Override (Optional)

```python
# Override motor specs from XML if needed
from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.motor_database import load_motor_spec

robot = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file("unitree_g1.xml"),
    articulation=EntityArticulationInfoCfg(
        actuators=(
            # Override specific actuators
            ElectricalMotorActuatorCfg(
                target_names_expr=("left_hip_actuator",),
                motor_spec=load_motor_spec("custom_calibrated_motor"),
            ),
        ),
    ),
)
```

## Implementation Plan

### Phase 2 Extensions

#### 1. XML Attribute Support

**Add custom attribute handling in MuJoCo spec editing:**

```python
# In ElectricalMotorActuatorCfg.edit_spec()
def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    motor = self.motor_spec

    for target_name in target_names:
        # Find actuator in spec
        actuator = find_actuator(spec, target_name)

        # Apply motor parameters
        actuator.gear[0] = motor.gear_ratio
        actuator.forcerange = [0, motor.peak_torque]

        # Write motor_spec as custom attribute (for saving XML)
        # MuJoCo supports actuator.info for custom metadata
        actuator.info = f"motor_spec:{motor.motor_id}"
```

**Note**: MuJoCo's `actuator.info` field is a string that gets written to XML as an attribute.

#### 2. XML Parsing for Motor Specs

**Add detection in Entity initialization:**

```python
# In Entity.__init__ or Entity.initialize()
def _parse_motor_specs_from_xml(self, spec: mujoco.MjSpec) -> list[tuple[str, str]]:
    """Parse motor_spec attributes from XML actuators.

    Returns:
        List of (actuator_name, motor_id) tuples
    """
    motor_specs = []

    for actuator in spec.actuators:
        # Check actuator.info for motor_spec:motor_id format
        if actuator.info and actuator.info.startswith("motor_spec:"):
            motor_id = actuator.info.split(":", 1)[1]
            motor_specs.append((actuator.name, motor_id))

    return motor_specs
```

#### 3. Auto-Actuator Creation

**Create ElectricalMotorActuator for detected motor specs:**

```python
def _create_auto_actuators(self, motor_specs: list[tuple[str, str]]) -> list[Actuator]:
    """Create electrical actuators for motor specs found in XML."""
    from mjlab.actuator import ElectricalMotorActuatorCfg
    from mjlab.motor_database import load_motor_spec

    auto_actuators = []

    for actuator_name, motor_id in motor_specs:
        # Load motor spec from database
        motor = load_motor_spec(motor_id)

        # Create electrical actuator config
        cfg = ElectricalMotorActuatorCfg(
            target_names_expr=(actuator_name,),
            motor_spec=motor,
        )

        # Build actuator
        actuator = cfg.build(self, [actuator_id], [actuator_name])
        auto_actuators.append(actuator)

    return auto_actuators
```

#### 4. Integration Point

**In Entity initialization flow:**

```python
def __init__(self, cfg: EntityCfg, ...):
    # 1. Load XML spec
    spec = cfg.spec_fn()

    # 2. Parse motor specs from XML
    motor_specs = self._parse_motor_specs_from_xml(spec)

    # 3. Create auto-actuators
    auto_actuators = self._create_auto_actuators(motor_specs)

    # 4. Merge with user-specified actuators (user overrides XML)
    all_actuators = auto_actuators + user_actuators

    # 5. Continue normal initialization...
```

### MuJoCo Custom Attribute Options

MuJoCo supports custom data in multiple ways:

**Option 1: Use `actuator.info` (string field)**
```xml
<motor name="hip" joint="hip_joint" info="motor_spec:unitree_7520_14"/>
```
- Pros: Simple, string-based, gets written/read automatically
- Cons: Not structured, need to parse string

**Option 2: Use custom numeric/text data**
```xml
<motor name="hip" joint="hip_joint">
  <custom key="motor_spec" value="unitree_7520_14"/>
</motor>
```
- Pros: Structured, MuJoCo's official custom data mechanism
- Cons: Slightly more verbose

**Recommendation**: Start with `info` field (simpler), migrate to `custom` if needed.

## Benefits

### For Researchers
1. **Share complete robot models** - XML includes motor specs
2. **Automatic electrical simulation** - No Python configuration needed
3. **MuJoCo Menagerie integration** - Motor specs alongside robot models

### For Reproducibility
1. **Self-documenting XMLs** - Motor specs visible in XML
2. **Version control friendly** - One file contains everything
3. **Easy comparison** - See which motors different researchers used

### For Sim-to-Real
1. **Calibrated specs per robot** - Each robot serial number can have calibrated XML
2. **Easy swapping** - Change motor_spec in XML to test different motors
3. **Hardware matching** - Simulation matches real robot configuration

## Backward Compatibility

- XMLs without `motor_spec` work as before (traditional actuators)
- Can mix electrical and traditional actuators in same robot
- Python override still works for custom motor specs
- No breaking changes to existing code

## Example: MuJoCo Menagerie Integration

```bash
mujoco_menagerie/
├── unitree_g1/
│   ├── g1.xml                    # Main robot XML
│   ├── scene.xml                 # Scene setup
│   └── motors/                   # Motor specs directory
│       ├── g1_7520_14.json      # Hip motor spec
│       └── g1_5020_9.json       # Ankle motor spec
```

**g1.xml:**
```xml
<mujoco model="unitree_g1">
  <compiler meshdir="assets"/>

  <actuator>
    <!-- Hip motors - electrical simulation enabled -->
    <motor name="left_hip_pitch" joint="left_hip_pitch_joint"
           gear="14.5" motor_spec="g1_7520_14"/>
    <motor name="right_hip_pitch" joint="right_hip_pitch_joint"
           gear="14.5" motor_spec="g1_7520_14"/>

    <!-- Ankle motors - electrical simulation enabled -->
    <motor name="left_ankle" joint="left_ankle_joint"
           gear="9.0" motor_spec="g1_5020_9"/>
    <motor name="right_ankle" joint="right_ankle_joint"
           gear="9.0" motor_spec="g1_5020_9"/>
  </actuator>
</mujoco>
```

**Usage:**
```python
from mjlab.entity import EntityCfg
from mjlab.motor_database import add_motor_database_path

# Add Menagerie motors to search path
add_motor_database_path("~/mujoco_menagerie/unitree_g1/motors")

# Load robot - motors auto-detected!
robot = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file("~/mujoco_menagerie/unitree_g1/g1.xml")
)

# Electrical simulation runs automatically
# Access motor electrical state during simulation:
# - actuator.current, actuator.voltage, actuator.temperature
```

## Implementation Checklist

### Phase 2 (Electrical Actuator)
- [ ] Add `motor_spec` writing in `ElectricalMotorActuatorCfg.edit_spec()`
- [ ] Add XML parsing for `motor_spec` attributes
- [ ] Add auto-actuator creation logic
- [ ] Integrate with Entity initialization
- [ ] Add tests for XML round-trip (write motor_spec, read back)
- [ ] Add tests for auto-loading
- [ ] Document XML format in actuator docs

### Phase 3 (MuJoCo Menagerie)
- [ ] Create PR to add motor specs to Menagerie robots
- [ ] Add motor search path auto-detection for Menagerie
- [ ] Document Menagerie integration workflow
- [ ] Create example showing Menagerie robot with electrical simulation

## Open Questions

1. **Multiple motor spec formats**: Should we support versioned motor specs (e.g., `motor_spec="unitree_7520_14:v2"`)?
2. **Fallback behavior**: What if motor_spec references non-existent motor? Error or warn and use default?
3. **Per-robot calibration**: How to handle robot-specific calibrated motor specs (e.g., `motor_spec="unitree_7520_14_serial_001"`)?

## Recommendation

Implement this in **Phase 2** alongside the electrical actuator, because:
1. Natural integration point when creating ElectricalMotorActuator
2. Enables immediate testing with complete XMLs
3. Makes Phase 2 deliverable more valuable (works with just XML!)
4. Sets up Phase 3 for Menagerie contribution

This design makes mjlab's electrical simulation **automatically available** to anyone using Menagerie robots with motor specs!
