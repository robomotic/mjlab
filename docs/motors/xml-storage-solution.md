# Motor and Battery Spec XML Storage: Solution

**Date**: 2026-03-23
**Status**: Tested & Verified

## Problem

We need to store motor_spec and battery_spec references in MuJoCo XML files such that:
1. Standard MuJoCo can load the XML without errors
2. The specs are preserved through write/read cycles
3. mjlab can auto-detect and load motor/battery specs

## MuJoCo XML Schema Constraints

**❌ Custom attributes NOT supported:**
```xml
<!-- This BREAKS - MuJoCo rejects unknown attributes -->
<motor name="motor1" joint="joint1" motor_spec="test_motor"/>
<text name="battery1" battery_spec="turnigy_6s2p_5000mah"/>
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

    <!-- Battery specification (scene-level) -->
    <text name="battery_main" data="battery_spec:turnigy_6s2p_5000mah"/>

    <!-- Optional: Map entities to batteries -->
    <text name="battery_entity_robot" data="battery:battery_main"/>
  </custom>
</mujoco>
```

### Benefits

1. ✅ **Standard MuJoCo compatible** - XML loads without errors
2. ✅ **Preserved through roundtrips** - `spec.to_xml()` includes custom text
3. ✅ **Accessible in Python** - Can iterate `spec.texts`
4. ✅ **Official MuJoCo feature** - Documented and supported
5. ✅ **Flexible** - Can store motor specs, battery specs, metadata, etc.

### Usage Pattern

**Motor specs:**
- Naming convention: `motor_{actuator_name}`
- Data format: `motor_spec:{motor_id}`

**Battery specs:**
- Naming convention: `battery_{battery_name}`
- Data format: `battery_spec:{battery_id}`

**Battery-entity mapping:**
- Naming convention: `battery_entity_{entity_name}`
- Data format: `battery:{battery_name}`

## Implementation

### Writing Motor Spec to XML

```python
def write_motor_spec_to_xml(spec: mujoco.MjSpec, actuator_name: str, motor_id: str):
    """Add motor_spec as custom text data."""
    text = spec.add_text()
    text.name = f"motor_{actuator_name}"
    text.data = f"motor_spec:{motor_id}"
```

### Writing Battery Spec to XML

```python
def write_battery_spec_to_xml(spec: mujoco.MjSpec, battery_name: str, battery_id: str):
    """Add battery_spec as custom text data."""
    text = spec.add_text()
    text.name = f"battery_{battery_name}"
    text.data = f"battery_spec:{battery_id}"
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

### Reading Battery Spec from XML

```python
def parse_battery_specs(spec: mujoco.MjSpec) -> dict[str, str]:
    """Parse battery specs from custom text elements.

    Returns:
        Dict mapping battery_name -> battery_id
    """
    battery_specs = {}

    for text in spec.texts:
        # Check if this is a battery_spec entry
        if text.name.startswith("battery_") and text.data.startswith("battery_spec:"):
            battery_name = text.name[8:]  # Remove "battery_" prefix
            battery_id = text.data.split(":", 1)[1]
            battery_specs[battery_name] = battery_id

    return battery_specs
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

def _create_battery_manager_from_xml(self, spec: mujoco.MjSpec):
    """Auto-create battery manager for battery specs in XML."""
    from mjlab.battery_database import load_battery_spec
    from mjlab.battery import BatteryManagerCfg

    # Parse battery specs from XML
    battery_specs = parse_battery_specs(spec)

    # Create battery manager (use first battery spec found)
    if battery_specs:
        battery_name, battery_id = next(iter(battery_specs.items()))
        battery_spec = load_battery_spec(battery_id)

        cfg = BatteryManagerCfg(
            battery_spec=battery_spec,
            entity_names=(self.name,),
        )

        return cfg

    return None
```

## Complete Example

### Robot XML with Motor and Battery Specs

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

  <!-- Motor and Battery specifications -->
  <custom>
    <!-- Motor specs for each actuator -->
    <text name="motor_left_hip_motor" data="motor_spec:unitree_7520_14"/>
    <text name="motor_left_knee_motor" data="motor_spec:unitree_5020_9"/>

    <!-- Battery specification (scene-level) -->
    <text name="battery_main" data="battery_spec:turnigy_6s2p_5000mah"/>

    <!-- Battery-entity mapping (which entity uses this battery) -->
    <text name="battery_entity_robot" data="battery:battery_main"/>
  </custom>
</mujoco>
```

### Loading in mjlab

```python
from mjlab.scene import SceneCfg
from mjlab.entity import EntityCfg

# Simple - motor and battery specs auto-detected from <custom><text> elements!
scene_cfg = SceneCfg(
    num_envs=4,
    entities={
        "robot": EntityCfg(
            spec_fn=lambda: mujoco.MjSpec.from_file("unitree_g1.xml")
        )
    }
)

# mjlab automatically:
# 1. Parses <custom><text name="motor_*"> elements
# 2. Extracts motor_spec:motor_id
# 3. Calls load_motor_spec(motor_id)
# 4. Creates ElectricalMotorActuator
# 5. Parses <custom><text name="battery_*"> elements
# 6. Extracts battery_spec:battery_id
# 7. Calls load_battery_spec(battery_id)
# 8. Creates BatteryManager at scene level
# 9. Battery powers motors with dynamic voltage feedback!
```

## MuJoCo Menagerie Integration

This approach is perfect for Menagerie:

```
mujoco_menagerie/
├── unitree_g1/
│   ├── g1.xml                    # Robot with <custom><text> motor and battery specs
│   ├── scene.xml
│   ├── motors/
│   │   ├── g1_7520_14.json      # Motor specs
│   │   └── g1_5020_9.json
│   └── batteries/
│       └── g1_battery.json       # Battery specs
```

Researchers can:
1. Add motor and battery specs to robot XMLs using `<custom><text>`
2. Share XMLs via Menagerie with complete electrical specifications
3. Anyone loading the XML gets automatic electrical simulation with power constraints
4. Battery voltage sag affects motor performance realistically

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

Use `<custom><text>` elements with the naming conventions:

**For motors:**
- **Name**: `motor_{actuator_name}`
- **Data**: `motor_spec:{motor_id}`

**For batteries:**
- **Name**: `battery_{battery_name}`
- **Data**: `battery_spec:{battery_id}`

**For battery-entity mapping:**
- **Name**: `battery_entity_{entity_name}`
- **Data**: `battery:{battery_name}`

This provides the cleanest, most compatible, and most maintainable solution for storing motor and battery specifications in MuJoCo XML files.

## Test Results

All tests pass with this approach:
- ✅ MuJoCo loads XML without errors
- ✅ Custom text preserved through roundtrips
- ✅ Programmatic access works
- ✅ Compatible with all actuator types
- ✅ Compatible with battery specifications
- ✅ Human-readable in XML
- ✅ Supports scene-level battery management

See `tests/test_motor_xml.py` and `tests/test_battery_database.py` for complete test suite.
