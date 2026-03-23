# Implementation Plan: Automatic Motor/Battery Integration

## Summary
Make motor and battery physics completely automatic during `scene.step()` by auto-discovering specs from XML and integrating updates into the simulation loop.

## File Changes Required

### 1. src/mjlab/entity/entity.py

Add auto-discovery of motors from XML custom elements:

```python
@dataclass
class EntityCfg:
    # ... existing fields ...

    # NEW: Auto-discovery flags
    auto_discover_motors: bool = True
    """Automatically discover motor specs from XML <custom><text> elements."""

    auto_discover_batteries: bool = False  # Batteries handled at Scene level
    """Reserved for future use."""

class Entity:
    def __init__(self, cfg: EntityCfg) -> None:
        self.cfg = cfg
        self._actuators: list[actuator.Actuator] = []
        self._build_spec()
        self._identify_joints()
        self._apply_spec_editors()

        # NEW: Auto-discover motors before manual actuator addition
        if cfg.auto_discover_motors and cfg.articulation is None:
            self._auto_discover_motors()

        self._add_actuators()
        self._add_initial_state_keyframe()

    def _auto_discover_motors(self) -> None:
        """Auto-discover motor specs from XML and create ElectricalMotorActuators."""
        from mjlab.motor_database import load_motor_spec
        from mjlab.motor_database.xml_integration import parse_motor_specs_from_xml
        from mjlab.actuator import ElectricalMotorActuatorCfg

        # Parse motor specs from XML
        motor_specs = parse_motor_specs_from_xml(self._spec)

        if not motor_specs:
            return  # No motors defined

        # Group actuators by motor spec for efficiency
        motor_groups = {}
        for actuator_name, motor_id in motor_specs.items():
            if motor_id not in motor_groups:
                motor_groups[motor_id] = []
            motor_groups[motor_id].append(actuator_name)

        # Create ElectricalMotorActuators for each motor type
        actuators = []
        for motor_id, actuator_names in motor_groups.items():
            motor_spec = load_motor_spec(motor_id)

            # Create regex pattern matching all actuators with this motor
            pattern = f"({'|'.join(actuator_names)})"

            actuator_cfg = ElectricalMotorActuatorCfg(
                target_names_expr=(pattern,),
                motor_spec=motor_spec,
                # Use sensible defaults (can be overridden)
                stiffness=200.0,
                damping=10.0,
                saturation_effort=motor_spec.peak_torque,
                velocity_limit=motor_spec.no_load_speed,
            )
            actuators.append(actuator_cfg)

        # Update config with auto-discovered actuators
        self.cfg.articulation = EntityArticulationInfoCfg(
            actuators=tuple(actuators)
        )
```

### 2. src/mjlab/scene/scene.py

Add auto-discovery of battery and automatic step integration:

```python
@dataclass
class SceneCfg:
    # ... existing fields ...

    # NEW: Auto-battery configuration
    auto_battery: bool = True
    """Automatically discover and create battery from XML specs."""

    battery: BatteryManagerCfg | None = None
    """Manual battery config (overrides auto-discovery)."""

class Scene:
    def __init__(self, cfg: SceneCfg, device: str = "cpu"):
        # ... existing initialization ...

        # NEW: Auto-discover and create battery
        self._battery_manager: BatteryManager | None = None
        if cfg.battery or cfg.auto_battery:
            self._setup_battery(cfg)

    def _setup_battery(self, cfg: SceneCfg) -> None:
        """Auto-discover or create battery from config."""
        from mjlab.battery_database import load_battery_spec
        from mjlab.battery_database.xml_integration import parse_battery_specs_from_xml

        battery_cfg = cfg.battery

        if battery_cfg is None and cfg.auto_battery:
            # Auto-discover from any entity's spec
            for entity in self._entities.values():
                battery_specs = parse_battery_specs_from_xml(entity._spec)
                if battery_specs:
                    # Found battery specs - create config
                    battery_id = list(battery_specs.values())[0]
                    battery_spec = load_battery_spec(battery_id)

                    battery_cfg = BatteryManagerCfg(
                        battery_spec=battery_spec,
                        entity_names=tuple(self._entities.keys()),
                        initial_soc=1.0,
                        enable_voltage_feedback=True,
                    )
                    break

        if battery_cfg:
            self._battery_manager = BatteryManager(battery_cfg, self)
            self._battery_manager.initialize(
                num_envs=cfg.num_envs,
                device=self._device
            )

    @property
    def battery(self) -> BatteryManager | None:
        """Access battery manager."""
        return self._battery_manager

    def step(self) -> None:
        """Step simulation with automatic motor/battery updates."""
        # NEW: Pre-step motor/battery updates
        self._pre_step_actuators()

        # Existing: Step physics
        self._sim.step()

        # NEW: Post-step updates
        self._post_step_actuators()

    def _pre_step_actuators(self) -> None:
        """Update actuators before physics step."""
        # Let actuators read MuJoCo state and compute currents
        for entity in self._entities.values():
            for act in entity._actuators:
                if hasattr(act, 'pre_step'):
                    act.pre_step(self._sim.model, self._sim.data)

        # Aggregate motor currents to battery
        if self._battery_manager:
            self._battery_manager.aggregate_current()
            self._battery_manager.update(self._sim.model.opt.timestep)
            self._battery_manager.compute_voltage()

            # Apply voltage feedback to motors
            if self._battery_manager.cfg.enable_voltage_feedback:
                for entity in self._entities.values():
                    for act in entity._actuators:
                        if hasattr(act, 'apply_voltage_feedback'):
                            act.apply_voltage_feedback(self._battery_manager.voltage)

    def _post_step_actuators(self) -> None:
        """Update actuators after physics step."""
        for entity in self._entities.values():
            for act in entity._actuators:
                if hasattr(act, 'post_step'):
                    act.post_step(self._sim.model, self._sim.data)
```

### 3. src/mjlab/actuator/electrical_motor.py

Add step integration methods:

```python
class ElectricalMotorActuator(Actuator):
    # ... existing code ...

    def pre_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called before physics step - calculate motor current from torque."""
        # Read actuator forces from MuJoCo
        for i, act_idx in enumerate(self._target_actuator_indices):
            # Get actuator force (this is the torque/force being applied)
            torque = abs(data.actuator_force[act_idx])

            # Calculate motor current from torque
            motor_spec = self.cfg.motor_spec
            if torque < 1e-6:
                current = motor_spec.no_load_current
            else:
                torque_clamped = min(torque, motor_spec.peak_torque)
                current = motor_spec.no_load_current + \
                         (motor_spec.stall_current - motor_spec.no_load_current) * \
                         (torque_clamped / motor_spec.stall_torque)

            self.current[i] = current

    def apply_voltage_feedback(self, battery_voltage: torch.Tensor) -> None:
        """Apply battery voltage scaling to motor output."""
        # Scale motor torque based on battery voltage
        voltage_ratio = battery_voltage / self.cfg.motor_spec.nominal_voltage

        # Voltage below nominal reduces available torque
        if voltage_ratio < 1.0:
            self.cfg.saturation_effort *= voltage_ratio.item()

    def post_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called after physics step - update metrics."""
        # Update motor state metrics (temperature, efficiency, etc.)
        pass
```

### 4. NEW: src/mjlab/robot.py

Create convenience API for ultra-simple usage:

```python
"""High-level robot controller with automatic motor/battery integration."""

from __future__ import annotations
from typing import Any
import mujoco
from mjlab.scene import Scene, SceneCfg
from mjlab.entity import EntityCfg

def load_robot(
    model_path: str,
    device: str = "cpu",
    num_envs: int = 1,
    **kwargs: Any
) -> RobotController:
    """Load robot with automatic motor/battery discovery.

    This is the simplest API - motors and batteries are automatically
    discovered from XML custom elements and integrated into physics.

    Args:
        model_path: Path to MuJoCo XML file
        device: Device for computation ("cpu" or "cuda")
        num_envs: Number of parallel environments
        **kwargs: Additional scene configuration options

    Returns:
        RobotController with automatic metrics

    Example:
        >>> robot = load_robot("g1_with_motors_battery.xml")
        >>> robot.set_joint_positions({"left_shoulder_pitch": 0.5})
        >>> robot.step()
        >>> print(f"Battery: {robot.battery_soc * 100:.1f}%")
    """
    spec = mujoco.MjSpec.from_file(model_path)
    robot_name = spec.modelname or "robot"

    scene_cfg = SceneCfg(
        num_envs=num_envs,
        entities={
            robot_name: EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_file(model_path),
                auto_discover_motors=True,
            )
        },
        auto_battery=True,
        **kwargs
    )

    scene = Scene(scene_cfg, device=device)
    return RobotController(scene, robot_name)

class RobotController:
    """User-friendly wrapper with automatic motor/battery metrics."""

    def __init__(self, scene: Scene, entity_name: str):
        self._scene = scene
        self._entity_name = entity_name
        self._entity = scene.entities[entity_name]

    def set_joint_positions(self, targets: dict[str, float]) -> None:
        """Set joint position targets."""
        for joint_name, position in targets.items():
            joint_id = mujoco.mj_name2id(
                self._scene.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )
            if joint_id >= 0:
                self._scene.data.ctrl[joint_id] = position

    def step(self) -> None:
        """Step simulation - all motor/battery updates automatic."""
        self._scene.step()

    @property
    def battery_soc(self) -> float:
        """Battery state of charge (0-1)."""
        battery = self._scene.battery
        return battery.soc[0].item() if battery else 1.0

    @property
    def battery_voltage(self) -> float:
        """Battery terminal voltage in volts."""
        battery = self._scene.battery
        return battery.voltage[0].item() if battery else 0.0

    @property
    def battery_current(self) -> float:
        """Total battery current draw in amps."""
        battery = self._scene.battery
        return battery.current[0].item() if battery else 0.0

    @property
    def battery_temperature(self) -> float:
        """Battery temperature in Celsius."""
        battery = self._scene.battery
        return battery.temperature[0].item() if battery else 25.0

    @property
    def motor_currents(self) -> dict[str, float]:
        """Current draw for each motor in amps."""
        from mjlab.actuator import ElectricalMotorActuator
        currents = {}
        for act in self._entity._actuators:
            if isinstance(act, ElectricalMotorActuator):
                for i, name in enumerate(act.names):
                    currents[name] = act.current[i].item()
        return currents
```

### 5. Update notebooks/humanoid_motor_demo_easy.ipynb

Replace manual controller with automatic API:

```python
# OLD Cell (manual calculation)
class SimpleRobotController:
    def move_joints(self, joint_targets, dt):
        # Manual current calculation...
        pass

# NEW Cell (automatic!)
from mjlab import load_robot

# Everything auto-discovered!
robot = load_robot("model/g1_with_motors_battery.xml")

print(f"✓ Robot loaded with automatic motor/battery integration")
print(f"  Motors: {len(robot.motor_currents)}")
print(f"  Battery: {robot.battery_soc * 100:.1f}% charged")
```

```python
# OLD simulation loop
for step in range(num_steps):
    joint_targets = {...}
    metrics = robot.move_joints(joint_targets, dt)

# NEW simulation loop (cleaner!)
for step in range(num_steps):
    robot.set_joint_positions({
        'left_shoulder_pitch_joint': shoulder_angle,
        'left_elbow_joint': elbow_angle,
    })

    # Everything automatic during step!
    robot.step()

    # Record metrics (auto-calculated!)
    if step % 5 == 0:
        soc_history.append(robot.battery_soc)
        current_history.append(robot.battery_current)
        voltage_history.append(robot.battery_voltage)
```

## Testing Strategy

### Test 1: Auto-Discovery
```python
def test_auto_discover_motors():
    """Test automatic motor discovery from XML."""
    cfg = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file("g1_with_motors_battery.xml"),
        auto_discover_motors=True,
    )

    entity = Entity(cfg)

    # Should auto-create actuators
    assert len(entity._actuators) > 0
    assert any(isinstance(a, ElectricalMotorActuator) for a in entity._actuators)
```

### Test 2: Automatic Battery
```python
def test_auto_battery():
    """Test automatic battery creation."""
    scene = Scene(SceneCfg(
        entities={
            "g1": EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_file("g1_with_motors_battery.xml"),
                auto_discover_motors=True,
            )
        },
        auto_battery=True,
    ))

    assert scene.battery is not None
    assert scene.battery.spec.capacity_ah == 9.0
```

### Test 3: Automatic Step Integration
```python
def test_automatic_step():
    """Test that motors/battery update automatically during step."""
    robot = load_robot("g1_with_motors_battery.xml")

    initial_soc = robot.battery_soc

    # Send commands
    robot.set_joint_positions({"left_shoulder_pitch_joint": 0.5})

    # Step multiple times
    for _ in range(100):
        robot.step()

    # Battery should have drained (automatic!)
    assert robot.battery_soc < initial_soc

    # Motor currents should be tracked (automatic!)
    assert len(robot.motor_currents) > 0
    assert sum(robot.motor_currents.values()) > 0
```

## Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Add `auto_discover_motors` to EntityCfg
- [ ] Implement `_auto_discover_motors()` in Entity
- [ ] Add step integration methods to ElectricalMotorActuator
- [ ] Tests for auto-discovery

### Week 2: Scene Integration
- [ ] Add `auto_battery` to SceneCfg
- [ ] Implement `_setup_battery()` in Scene
- [ ] Update `Scene.step()` with pre/post actuator updates
- [ ] Tests for automatic battery

### Week 3: High-Level API
- [ ] Create `src/mjlab/robot.py` with `load_robot()`
- [ ] Implement `RobotController` wrapper
- [ ] Add property accessors for metrics
- [ ] Tests for convenience API

### Week 4: Documentation & Examples
- [ ] Update notebooks to use new API
- [ ] Write migration guide
- [ ] Update getting started docs
- [ ] Create video tutorial

## Migration Path

### Current users (manual):
```python
# Still works!
battery_mgr = BatteryManager(...)
battery_mgr.aggregate_current()
battery_mgr.update(dt)
```

### New users (automatic):
```python
# Much simpler!
robot = load_robot("model.xml")
robot.step()  # Everything happens automatically
```

### Advanced users (hybrid):
```python
# Can still customize while using auto-discovery
scene = Scene(SceneCfg(
    entities={
        "robot": EntityCfg(
            spec_fn=...,
            auto_discover_motors=True,  # Auto motors
            articulation=EntityArticulationInfoCfg(
                actuators=(...,)  # Plus custom actuators
            )
        )
    },
    auto_battery=True,  # Auto battery
))
```

## Benefits Summary

1. **Zero Manual Physics**: Users don't calculate current/voltage
2. **Single Step Call**: Everything updates in `scene.step()`
3. **Discoverable**: Load XML and it just works
4. **Correct by Default**: No manual calculation errors
5. **Backwards Compatible**: Existing explicit API still works
6. **Performance**: Automatic batching and optimization
