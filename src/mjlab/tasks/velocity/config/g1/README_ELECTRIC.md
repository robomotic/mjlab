# Unitree G1 with Electrical Motors & Battery Visualization

Two new environments that extend the standard G1 velocity tasks with realistic electrical motors and battery management:

- `Mjlab-Velocity-Flat-Unitree-G1-Electric` - Flat terrain with electrical metrics
- `Mjlab-Velocity-Rough-Unitree-G1-Electric` - Rough terrain with electrical metrics

## Quick Start

### Run with Zero Policy (No Training Required)

```bash
# Launch with Viser web viewer - metrics will appear automatically
uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric --agent zero --viewer viser

# Or with random actions
uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric --agent random --viewer viser
```

The Viser viewer will:
1. Open in your browser at `http://localhost:8080`
2. Display the **Metrics tab** with 10 electrical metrics:
   - Motor metrics: current (A), voltage (V), power (W), temperature (°C), back-EMF (V)
   - Battery metrics: SOC (%), voltage (V), current (A), power (W), temperature (°C)
3. Show **real-time plots** with 300-point history
4. Allow **filtering** with checkboxes and text search

### Run with Trained Policy

```bash
# Use a trained checkpoint from W&B
uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric --viewer viser \
    --wandb-run-path your-entity/your-project/run_id
```

## What's Different from Standard G1?

### 1. **Realistic Motor Models**
- Hip/knee joints: Unitree 7520-14 (88 N⋅m continuous torque)
- Ankle/arm joints: Unitree 5020-9 (20 N⋅m continuous torque)
- Accurate electrical properties (resistance, inductance, motor constants)
- Thermal modeling (winding temperature rises with load)

### 2. **Battery System**
- Unitree G1 9Ah Li-ion battery (199.8 Wh, 21.6V nominal)
- Tracks state of charge (SOC) - depletes during simulation
- Voltage feedback: motor performance degrades as battery drains
- Temperature modeling
- **Regenerative braking disabled by default** (realistic for Li-ion batteries)

### 3. **Real-time Metrics in Viser**
10 aggregate metrics automatically visualized:
- `motor_current_avg` - Average current across all motors
- `motor_voltage_avg` - Average voltage across all motors
- `motor_power_total` - Total power dissipation (I²R losses)
- `motor_temperature_max` - Hottest motor winding
- `motor_back_emf_avg` - Average back-EMF
- `battery_soc` - State of charge (0-1 scale)
- `battery_voltage` - Terminal voltage
- `battery_current` - Total current draw **from motors only**
- `battery_power` - Output power **to motors only**
- `battery_temperature` - Battery temperature

**Important: Interpreting Battery Metrics**

Battery current and power represent **only the power drawn by motors**. When you see zero current/power:
- ✅ **Simulation perspective**: Motors are not drawing power (idle or backdriven with regen disabled)
- ⚠️ **Real robot perspective**: Battery would still supply baseline power for:
  - Control electronics (motor drivers, microcontrollers: ~2-5A)
  - Sensors (IMU, cameras, encoders: ~0.5-1A)
  - Communication systems (WiFi, Ethernet: ~0.5-1A)
  - Computation (onboard computer: ~2-5A)
  - **Typical baseline: 5-10A even when standing still**

The current simulation models only motor power draw. In a real Unitree G1, battery current would never drop below ~5-10A during operation.

## Training with Electrical Metrics

Train a new policy that accounts for battery drain and motor dynamics:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1-Electric \
    --num-envs 4096 \
    --wandb-project g1-electric
```

The trained policy will learn to:
- Manage energy consumption (battery SOC decreases over episode)
- Adapt to voltage sag (motors weaken as battery drains)
- Account for motor thermal limits (temperature increases with use)

## Customization

Edit `/src/mjlab/tasks/velocity/config/g1/env_cfgs_electric.py` to:

### Add Per-Joint Metrics

Track individual joint currents for debugging:

```python
from mjlab.envs.mdp.metrics import motor_current_joint
from mjlab.managers import MetricsTermCfg

cfg.metrics.update({
    "left_knee_current": MetricsTermCfg(
        func=motor_current_joint,
        params={"joint_name": "left_knee"},
    ),
    "right_hip_current": MetricsTermCfg(
        func=motor_current_joint,
        params={"joint_name": "right_hip_pitch"},
    ),
})
```

### Add Cumulative Energy Tracking

Track total energy consumed per episode:

```python
from mjlab.envs.mdp.metrics import CumulativeEnergyMetric

cfg.metrics.update({
    "energy_consumed_wh": MetricsTermCfg(func=CumulativeEnergyMetric()),
})
```

This resets to zero at episode start and logs final value for analysis.

### Use Different Motor Models

Replace motors from the database:

```python
from mjlab.motor_database import load_motor_spec

# Use different motor for all joints
custom_motor = load_motor_spec("your_motor_id")
```

### Use Different Battery

Replace battery from the database:

```python
from mjlab.battery_database import load_battery_spec

# Use different battery
custom_battery = load_battery_spec("your_battery_id")
```

## Viewing Metrics

### In Viser (Web Browser)
1. Metrics appear automatically in the **Metrics tab**
2. Enable/disable plots with checkboxes
3. Search/filter with text box
4. Real-time updates (60 fps)
5. 300-point history per metric

### Programmatic Access

```python
# During training/evaluation
env = ManagerBasedRlEnv(cfg, device="cuda")
env.step(actions)

# Get metrics for environment 0
metrics = env.metrics_manager.get_active_iterable_terms(env_idx=0)
metrics_dict = {name: values[0] for name, values in metrics}

print(f"Battery SOC: {metrics_dict['battery_soc'] * 100:.1f}%")
print(f"Motor current: {metrics_dict['motor_current_avg']:.2f}A")
```

## Performance Notes

- **Training overhead**: <2% compared to standard actuators
- **Visualization overhead**: Negligible (metrics computed on-device)
- **Memory**: Same as standard G1 environments

## Troubleshooting

### Metrics not appearing in Viser
- Ensure you're using `--viewer viser` (not `native`)
- Check browser console for errors
- Verify metrics in code: `print(cfg.metrics.keys())`

### Battery drains too fast/slow
- Adjust `initial_soc` in `BatteryManagerCfg`
- Use larger battery: `load_battery_spec("custom_larger_battery")`
- Adjust motor `effort_limit` (lower = less power draw)

### Motors too weak/strong
- Adjust PD gains: `stiffness`, `damping` in `ElectricalMotorActuatorCfg`
- Use different motor models from database
- Tune `effort_limit` (continuous torque rating)

## Cable-Powered Mode (Infinite Power)

For robots powered by a cable (wall power) instead of a battery, use the cable-powered configuration:

```bash
# Launch cable-powered G1 (no battery, infinite power)
uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric-Cable --agent zero --viewer viser
```

### What is Cable-Powered Mode?

Cable-powered mode provides **electrical motors WITHOUT a battery**, giving:
- ✅ Full rated motor voltage (24V) at all times
- ✅ No voltage sag under load
- ✅ No SOC depletion
- ✅ No power constraints
- ❌ No energy tracking (no battery metrics)

### When to Use Cable-Powered Mode

**Use cable power for:**
- **Benchtop testing** with wall power supply
- **Training without power constraints** (maximum performance)
- **Performance evaluation** at full motor capability
- **Debugging motor behavior** without battery complexity

**Use battery power for:**
- **Realistic simulation** of untethered robots
- **Energy efficiency** optimization and tracking
- **Endurance testing** with battery depletion
- **Training policies** that adapt to voltage sag

### Configuration Comparison

| Feature | Battery-Powered | Cable-Powered | Battery (feedback disabled) |
|---------|----------------|---------------|----------------------------|
| Voltage | Variable (sags under load) | Constant (24V) | Constant (but battery present) |
| SOC Tracking | ✅ Yes | ❌ No | ✅ Yes (but not used) |
| Energy Metrics | ✅ Battery + Motor | ✅ Motor Only | ✅ Battery + Motor |
| Performance | Degrades as battery drains | Constant (maximum) | Constant (maximum) |
| Use Case | Realistic simulation | Benchtop/training | Energy accounting only |

### How It Works

When no battery is configured (`cfg.scene.battery = None`):
1. Motors initialize with their rated voltage from `motor_spec.voltage_range[1]`
2. Voltage never updated by battery → always at maximum
3. No voltage sag → no performance degradation
4. Motors operate at full capability throughout simulation

**Code example:**
```python
from mjlab.envs.mdp.metrics import electrical_metrics_preset

def my_cable_powered_robot():
    cfg = base_robot_cfg()

    # Add electrical motors (from motor database or XML auto-discovery)
    cfg.scene.entities["robot"].articulation = EntityArticulationInfoCfg(
        actuators=(
            ElectricalMotorActuatorCfg(
                target_names_expr=(".*_joint",),
                motor_spec=load_motor_spec("unitree_7520_14"),
            ),
        )
    )

    # NO BATTERY CONFIGURED → Cable-powered mode
    # cfg.scene.battery = None  # (default, omit battery entirely)

    # Add motor metrics only (no battery metrics)
    cfg.metrics = electrical_metrics_preset(
        include_motor=True,
        include_battery=False,  # No battery present
    )

    return cfg
```

### Alternative: Battery with Feedback Disabled

If you want **energy tracking** but **no voltage limiting**:
```python
cfg.scene.battery = BatteryManagerCfg(
    battery_spec=load_battery_spec("unitree_g1_9ah"),
    entity_names=("robot",),
    initial_soc=1.0,
    enable_voltage_feedback=False,  # Track energy but don't limit voltage
)
```

This keeps battery metrics for logging but motors always have full voltage.

## Regenerative Braking Control

By default, batteries **do not accept regenerative braking** - when motors are backdriven (e.g., by gravity), negative current is clamped to zero. This is realistic for most commercial robot batteries (Li-Po, Li-ion) that lack charging circuits.

**Default configuration (no regenerative braking):**
```python
cfg.scene.battery = BatteryManagerCfg(
    battery_spec=load_battery_spec("unitree_g1_9ah"),
    entity_names=("robot",),
    initial_soc=1.0,
    enable_voltage_feedback=True,
    allow_regenerative_braking=False,  # Default: reject backfeed
)
```

When regenerative braking is **disabled** (default):
- Negative motor current is clamped to zero at battery level
- Energy dissipates as heat in motor windings (I²R loss)
- Battery SOC never increases
- Battery current and power are always non-negative
- More realistic for standard Li-Po/Li-ion batteries

**Enable regenerative braking (future batteries with charge controller):**
```python
cfg.scene.battery = BatteryManagerCfg(
    battery_spec=load_battery_spec("future_lifepo4_regen"),
    entity_names=("robot",),
    initial_soc=0.5,
    enable_voltage_feedback=True,
    allow_regenerative_braking=True,  # Allow energy return
)
```

When regenerative braking is **enabled**:
- Negative motor current flows back to battery
- Battery SOC increases during backdriving
- Battery current can be negative (charging)
- More power-efficient simulation
- Only use for battery specs with explicit charge acceptance capability

## Troubleshooting

### Battery current/power drops to zero

**What you're seeing:** Spikes where battery current and power go to zero in the metrics plots.

**Why this happens:**
- Battery metrics represent **only motor power draw**
- When all motors have low torque demand or are backdriven (with regen disabled), battery current → 0
- This is expected behavior in the simulation

**Real robot behavior:**
- Battery current would **never** be zero during operation
- Baseline power draw: 5-10A for electronics, sensors, computation, communication
- Motor power is typically 50-90% of total power during locomotion

**Current limitation:** The simulation models only motor electrical dynamics. Future enhancement could add a configurable `base_load_current` parameter to represent parasitic loads.

### Other Resources

- [Motor Database Documentation](../../../motor_database/)
- [Battery Database Documentation](../../../battery_database/)
- [Metrics Documentation](../../../envs/mdp/metrics.py)
- [Phase 6 Examples](../../../../examples/electrical_metrics_demo.py)
