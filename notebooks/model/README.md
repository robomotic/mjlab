# Unitree G1 Model with Motor and Battery Specifications

This directory contains a complete, self-contained model of the Unitree G1 humanoid robot
with motor and battery specifications embedded.

## Files

- `g1_with_motors_battery.xml` - Complete MuJoCo XML scene with motor and battery specs embedded
- `motors/` - Motor specification JSON files
  - `unitree_7520_14.json` - High-torque motor (used for hip joints)
  - `unitree_5020_9.json` - Medium-torque motor (used for other joints)
- `batteries/` - Battery specification JSON files
  - `unitree_g1_9ah.json` - Unitree G1 9Ah Li-ion battery

## Motor Mappings

### Unitree 7520-14 (High Torque)
Used for hip joints that require high torque:
- left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint
- right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint

### Unitree 5020-9 (Medium Torque)
Used for all other joints:
- Knee joints (left_knee_joint, right_knee_joint)
- Ankle joints (left/right_ankle_pitch/roll_joint)
- Waist joints (waist_yaw/roll/pitch_joint)
- Shoulder joints (left/right_shoulder_pitch/roll/yaw_joint)
- Elbow joints (left/right_elbow_joint)
- Wrist joints (left/right_wrist_roll/pitch/yaw_joint)

## Usage

Load the model with mjlab:

```python
import mujoco
from mjlab.motor_database import load_motor_spec
from mjlab.battery_database import load_battery_spec

# Load the XML
spec = mujoco.MjSpec.from_file("model/g1_with_motors_battery.xml")

# Motor and battery specs are embedded in the XML as custom text elements
# They can be parsed and used with mjlab's Scene and Entity APIs
```

## Specifications

### Robot
- Model: Unitree G1
- DOFs: 23 (6 per leg, 6 per arm, 3 waist, 1 head, 1 floating base)
- Height: 127cm
- Weight: 35kg

### Battery
- Capacity: 9000mAh (9Ah)
- Energy: 199.8Wh
- Nominal voltage: 21.6V (6S Li-ion)
- Chemistry: Li-ion
- Runtime: ~2 hours under normal operation

### Motors
- Hip motors: Unitree 7520-14 (30 Nm peak torque)
- Other motors: Unitree 5020-9 (20 Nm peak torque)

---

Generated with [mjlab](https://github.com/pdiprodi/mjlab) - GPU-accelerated robot simulation
