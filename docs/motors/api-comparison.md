# Motor/Battery API Evolution: Manual → Automatic

## Current State (Manual - Complex)

```python
# User must understand motor physics
def calculate_motor_current(torque, motor_spec):
    if abs(torque) < 1e-6:
        return motor_spec.no_load_current
    torque_clamped = min(abs(torque), motor_spec.peak_torque)
    return motor_spec.no_load_current + \
           (motor_spec.stall_current - motor_spec.no_load_current) * \
           (torque_clamped / motor_spec.stall_torque)

# User must manually aggregate currents
total_current = 0.0
for joint, torque in joint_torques.items():
    motor_spec = motor_specs[joint]
    current = calculate_motor_current(torque, motor_spec)
    total_current += current

# User must manually update battery
battery_mgr.current = torch.tensor([total_current])
battery_mgr.update(dt)
battery_mgr.compute_voltage()

# User must manually track metrics
metrics = {
    'soc': battery_mgr.soc[0].item(),
    'voltage': battery_mgr.voltage[0].item(),
    'current': battery_mgr.current[0].item(),
}
```

**Problems:**
- ❌ User must understand motor physics equations
- ❌ User must manually aggregate motor currents
- ❌ User must remember to update battery
- ❌ Easy to make mistakes (wrong order, missing updates)
- ❌ Verbose and error-prone

---

## Proposed State (Automatic - Simple)

```python
from mjlab import load_robot

# Load robot - motors/battery auto-discovered from XML
robot = load_robot("g1_with_motors_battery.xml")

# Just send position commands (standard MuJoCo API)
robot.set_joint_positions({
    'left_shoulder_pitch_joint': 0.5,
    'left_elbow_joint': 0.8,
})

# Everything automatic during step!
robot.step()  # ✅ Motors calculate current
              # ✅ Battery aggregates and updates
              # ✅ Voltage feedback applied
              # ✅ All metrics computed

# Access metrics directly (no manual calculation!)
print(f"Battery SOC: {robot.battery_soc * 100:.1f}%")
print(f"Voltage: {robot.battery_voltage:.2f}V")
print(f"Current: {robot.battery_current:.2f}A")
print(f"Motor currents: {robot.motor_currents}")
```

**Benefits:**
- ✅ Zero physics knowledge required
- ✅ Automatic current aggregation
- ✅ Automatic battery updates
- ✅ Impossible to forget updates
- ✅ Clean, simple API

---

## Side-by-Side: Complete Example

### Manual Approach (Current)
```python
# Load model
spec = mujoco.MjSpec.from_file("g1.xml")

# Parse motor specs manually
motor_specs_map = {}
motor_specs = parse_motor_specs_from_xml(spec)
for name, motor_id in motor_specs.items():
    motor_specs_map[name] = load_motor_spec(motor_id)

# Parse battery spec manually  
battery_specs = parse_battery_specs_from_xml(spec)
battery_id = list(battery_specs.values())[0]
battery_spec = load_battery_spec(battery_id)

# Create battery manager manually
battery_mgr = BatteryManager(
    BatteryManagerCfg(
        battery_spec=battery_spec,
        entity_names=("g1",),
        initial_soc=1.0,
    ),
    MockScene()
)
battery_mgr.initialize(num_envs=1, device="cpu")

# Simulation loop with manual updates
for step in range(1000):
    # Calculate torques manually
    torques = calculate_joint_torques(positions, velocities)
    
    # Calculate currents manually
    total_current = 0.0
    for joint_name, torque in torques.items():
        motor_spec = motor_specs_map[joint_name]
        current = calculate_motor_current(torque, motor_spec)
        total_current += current
    
    # Update battery manually
    battery_mgr.current = torch.tensor([total_current])
    battery_mgr.update(dt)
    battery_mgr.compute_voltage()
    
    # Apply voltage feedback manually
    voltage_ratio = battery_mgr.voltage / battery_spec.nominal_voltage
    # ... scale motor outputs ...
    
    # Record metrics manually
    soc_history.append(battery_mgr.soc[0].item())
    voltage_history.append(battery_mgr.voltage[0].item())
```

**Lines of code:** ~50-60 lines  
**Concepts to understand:** Motor physics, current calculation, battery dynamics, voltage feedback  
**Error opportunities:** Many (wrong order, missing updates, calculation mistakes)

---

### Automatic Approach (Proposed)
```python
from mjlab import load_robot

# Load robot with auto-discovery
robot = load_robot("g1_with_motors_battery.xml")

# Simulation loop - all updates automatic
for step in range(1000):
    # Send position commands (standard MuJoCo)
    robot.set_joint_positions({
        'left_shoulder_pitch_joint': 0.5,
        'right_shoulder_pitch_joint': 0.5,
    })
    
    # Everything automatic!
    robot.step()
    
    # Record metrics (auto-computed)
    soc_history.append(robot.battery_soc)
    voltage_history.append(robot.battery_voltage)
```

**Lines of code:** ~10-15 lines  
**Concepts to understand:** Just MuJoCo position commands  
**Error opportunities:** Minimal (everything automatic)

---

## What Happens Automatically

When you call `robot.step()`, here's what happens behind the scenes:

```
1. Pre-Step Phase:
   ├─ For each ElectricalMotorActuator:
   │  ├─ Read actuator force from MuJoCo data
   │  ├─ Calculate motor current from torque (I = f(τ, motor_spec))
   │  └─ Store current in actuator.current tensor
   │
   ├─ Battery Manager:
   │  ├─ Aggregate all motor currents
   │  ├─ Update battery state (SOC, temperature)
   │  ├─ Compute terminal voltage with I·R drop
   │  └─ Apply voltage feedback to motors
   │
   └─ Ready for physics step

2. Physics Step:
   └─ mujoco.mj_step(model, data)  # Standard MuJoCo

3. Post-Step Phase:
   ├─ Update motor metrics (efficiency, temperature)
   └─ Update data structures for user access
```

**User sees:** Just `robot.step()`  
**User gets:** All metrics automatically computed

---

## API Levels

We provide three API levels for different use cases:

### Level 1: Ultra-Simple (Beginners)
```python
from mjlab import load_robot

robot = load_robot("model.xml")
robot.set_joint_positions({...})
robot.step()
print(robot.battery_soc)
```
**Use when:** Learning, prototyping, simple demos

### Level 2: Scene API (Intermediate)
```python
from mjlab.scene import Scene, SceneCfg
from mjlab.entity import EntityCfg

scene = Scene(SceneCfg(
    entities={
        "robot": EntityCfg(
            spec_fn=lambda: spec,
            auto_discover_motors=True,
        )
    },
    auto_battery=True,
))

scene.step()
print(scene.battery.soc)
```
**Use when:** Multi-robot, custom configs, RL training

### Level 3: Manual (Advanced)
```python
from mjlab.battery import BatteryManager
from mjlab.actuator import ElectricalMotorActuator

# Explicit configuration and control
battery = BatteryManager(cfg, scene)
actuator = ElectricalMotorActuator(motor_cfg)

# Manual updates
actuator.compute_current()
battery.aggregate_current()
battery.update(dt)
```
**Use when:** Custom physics, research, debugging

---

## Migration Examples

### Example 1: Simple Demo

**Before:**
```python
# 20+ lines of setup code
battery_mgr = BatteryManager(...)
motor_specs = {...}

for step in range(1000):
    # 10+ lines of manual updates
    current = calculate_current(...)
    battery_mgr.update(...)
```

**After:**
```python
robot = load_robot("model.xml")

for step in range(1000):
    robot.step()
```

**Reduction:** 90% fewer lines

---

### Example 2: RL Training

**Before:**
```python
class RobotEnv:
    def step(self, action):
        # Manual motor/battery updates
        currents = [...]
        battery.update(...)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        return obs, reward, done, info
```

**After:**
```python
class RobotEnv:
    def step(self, action):
        self.robot.set_joint_positions(action)
        self.robot.step()  # Motors/battery automatic
        
        obs = self._get_obs()
        reward = self._compute_reward()
        return obs, reward, done, info
```

**Benefits:** Cleaner code, automatic metrics in `info` dict

---

## Summary: Why This Matters

### For Beginners
- **Before:** Must understand motor physics, current calculations, battery dynamics
- **After:** Just load model and send position commands
- **Impact:** Can start using realistic physics in minutes, not hours

### For Researchers  
- **Before:** Manual tracking error-prone, hard to ensure correctness
- **After:** Automatic updates guarantee correctness
- **Impact:** Focus on research questions, not implementation details

### For Engineers
- **Before:** Verbose code, many opportunities for bugs
- **After:** Clean, simple API with automatic validation
- **Impact:** Faster development, fewer bugs, easier maintenance

### For Everyone
- **Before:** Need deep understanding to use motor/battery physics
- **After:** Physics "just works" transparently
- **Impact:** Democratizes access to realistic robot simulation
