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

### 3. **Real-time Metrics in Viser**
10 aggregate metrics automatically visualized:
- `motor_current_avg` - Average current across all motors
- `motor_voltage_avg` - Average voltage across all motors
- `motor_power_total` - Total power dissipation (I²R losses)
- `motor_temperature_max` - Hottest motor winding
- `motor_back_emf_avg` - Average back-EMF
- `battery_soc` - State of charge (0-1 scale)
- `battery_voltage` - Terminal voltage
- `battery_current` - Total current draw
- `battery_power` - Output power
- `battery_temperature` - Battery temperature

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

## See Also

- [Motor Database Documentation](../../../motor_database/)
- [Battery Database Documentation](../../../battery_database/)
- [Metrics Documentation](../../../envs/mdp/metrics.py)
- [Phase 6 Examples](../../../../examples/electrical_metrics_demo.py)
