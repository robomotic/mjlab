# Motor Spec XML Storage: Solution

**Date**: 2026-03-23
**Status**: Tested & Verified

## Problem

We need to store motor_spec references in MuJoCo XML files such that:
1. Standard MuJoCo can load the XML without errors
2. The motor_spec is preserved through write/read cycles
3. mjlab can auto-detect and load motor specs

## MuJoCo XML Schema Constraints

**❌ Custom attributes NOT supported:**
```xml
<!-- This BREAKS - MuJoCo rejects unknown attributes -->
<motor name="motor1" joint="joint1" motor_spec="test_motor"/>
```
Error: `ValueError: XML Error: Schema violation: unrecognized attribute`

**❌ The `info` field is NOT written to XML:**
```python
actuator.info = "motor_spec:test_motor"  # Can set programmatically
# But spec.to_xml() does NOT include info in output!
```

## ✅ Solution: Use `<custom><text>` Elements

MuJoCo officially supports custom text data in `<custom>` section:

```xml
<mujoco model="unitree_g1">
  <worldbody>
    <body name="body1">
      <geom type="box" size="0.1 0.1 0.1"/>
      <joint name="hip_joint" type="hinge"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_hip" joint="hip_joint" gear="14.5"/>
  </actuator>

  <!-- Motor specifications stored in custom text data -->
  <custom>
    <text name="motor_left_hip" data="motor_spec:unitree_7520_14"/>
  </custom>
</mujoco>
```

### Benefits

1. ✅ **Standard MuJoCo compatible** - XML loads without errors
2. ✅ **Preserved through roundtrips** - `spec.to_xml()` includes custom text
3. ✅ **Accessible in Python** - Can iterate `spec.texts`
4. ✅ **Official MuJoCo feature** - Documented and supported
5. ✅ **Flexible** - Can store multiple motor specs, metadata, etc.

### Usage Pattern

**Naming convention:** `motor_{actuator_name}`
**Data format:** `motor_spec:{motor_id}`

## Implementation

### Writing Motor Spec to XML

```python
def write_motor_spec_to_xml(spec: mujoco.MjSpec, actuator_name: str, motor_id: str):
    """Add motor_spec as custom text data."""
    text = spec.add_text()
    text.name = f"motor_{actuator_name}"
    text.data = f"motor_spec:{motor_id}"
```

### Reading Motor Spec from XML

```python
def parse_motor_specs(spec: mujoco.MjSpec) -> dict[str, str]:
    """Parse motor specs from custom text elements.

    Returns:
        Dict mapping actuator_name -> motor_id
    """
    motor_specs = {}

    for text in spec.texts:
        # Check if this is a motor_spec entry
        if text.name.startswith("motor_") and text.data.startswith("motor_spec:"):
            actuator_name = text.name[6:]  # Remove "motor_" prefix
            motor_id = text.data.split(":", 1)[1]
            motor_specs[actuator_name] = motor_id

    return motor_specs
```

### Auto-Loading in Entity

```python
def _create_electrical_actuators_from_xml(self, spec: mujoco.MjSpec):
    """Auto-create electrical actuators for motor specs in XML."""
    from mjlab.motor_database import load_motor_spec
    from mjlab.actuator import ElectricalMotorActuatorCfg

    # Parse motor specs from XML
    motor_specs = parse_motor_specs(spec)

    # Create actuators
    auto_actuators = []
    for actuator_name, motor_id in motor_specs.items():
        motor = load_motor_spec(motor_id)

        cfg = ElectricalMotorActuatorCfg(
            target_names_expr=(actuator_name,),
            motor_spec=motor,
        )

        actuator = cfg.build(...)
        auto_actuators.append(actuator)

    return auto_actuators
```

## Complete Example

### Robot XML with Motor Specs

```xml
<mujoco model="unitree_g1">
  <worldbody>
    <body name="torso">
      <body name="left_hip">
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0.15 0"/>
        <joint name="left_hip_pitch" axis="0 1 0" range="-1.57 1.57"/>

        <body name="left_thigh">
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3"/>
          <joint name="left_knee" axis="0 1 0" range="0 2.7"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_hip_motor" joint="left_hip_pitch" gear="14.5"/>
    <motor name="left_knee_motor" joint="left_knee" gear="9.0"/>
  </actuator>

  <!-- Motor specifications -->
  <custom>
    <text name="motor_left_hip_motor" data="motor_spec:unitree_7520_14"/>
    <text name="motor_left_knee_motor" data="motor_spec:unitree_5020_9"/>
  </custom>
</mujoco>
```

### Loading in mjlab

```python
from mjlab.entity import EntityCfg

# Simple - motor specs auto-detected from <custom><text> elements!
robot = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file("unitree_g1.xml")
)

# mjlab automatically:
# 1. Parses <custom><text name="motor_*"> elements
# 2. Extracts motor_spec:motor_id
# 3. Calls load_motor_spec(motor_id)
# 4. Creates ElectricalMotorActuator
# 5. Electrical simulation runs!
```

## MuJoCo Menagerie Integration

This approach is perfect for Menagerie:

```
mujoco_menagerie/
├── unitree_g1/
│   ├── g1.xml                    # Robot with <custom><text> motor specs
│   ├── scene.xml
│   └── motors/
│       ├── g1_7520_14.json      # Motor specs
│       └── g1_5020_9.json
```

Researchers can:
1. Add motor specs to robot XMLs using `<custom><text>`
2. Share XMLs via Menagerie
3. Anyone loading the XML gets automatic electrical simulation

## Alternative Approaches (NOT Recommended)

### 1. User Data (Numeric Only)
```xml
<motor name="motor1" joint="joint1" user="689819847"/>
```
- ❌ Only stores numbers, not strings
- ❌ Needs lookup table to map numbers to motor IDs
- ❌ Not human-readable

### 2. Comments (Not Accessible)
```xml
<!-- motor_spec: unitree_7520_14 -->
<motor name="motor1" joint="joint1"/>
```
- ❌ Comments are stripped during parsing
- ❌ Not accessible programmatically

### 3. Actuator Name Encoding
```xml
<motor name="motor1_unitree_7520_14" joint="joint1"/>
```
- ❌ Pollutes actuator names
- ❌ Not clean separation of concerns

## Recommendation

Use `<custom><text>` elements with the naming convention:
- **Name**: `motor_{actuator_name}`
- **Data**: `motor_spec:{motor_id}`

This provides the cleanest, most compatible, and most maintainable solution for storing motor specifications in MuJoCo XML files.

## Test Results

All tests pass with this approach:
- ✅ MuJoCo loads XML without errors
- ✅ Custom text preserved through roundtrips
- ✅ Programmatic access works
- ✅ Compatible with all actuator types
- ✅ Human-readable in XML

See `tests/test_motor_xml.py` for complete test suite.
