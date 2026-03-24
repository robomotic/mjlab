"""Unitree G1 velocity environment with electrical motor/battery metrics.

This configuration extends the standard G1 velocity task with:
- Electrical motor actuators (realistic motor models)
- Battery manager (tracks SOC, voltage, current)
- Real-time electrical metrics visualization in Viser viewer

Register this task and run with:
  uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric --agent zero --viewer viser
"""

from mjlab.actuator import ElectricalMotorActuatorCfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.battery import BatteryManagerCfg
from mjlab.battery_database import load_battery_spec
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.metrics import electrical_metrics_preset
from mjlab.motor_database import load_motor_spec
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def unitree_g1_flat_electric_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity config with electrical motors/battery.

  This extends the standard G1 flat environment with:
  - Realistic electrical motor actuators for all joints
  - Battery manager tracking energy consumption
  - 15 electrical metrics for real-time monitoring:
    * 10 aggregate metrics (motor avg, battery stats)
    * 5 per-joint metrics (optional, enable in code below)
    * 2 cumulative metrics (energy consumed, work output)

  Args:
    play: Whether this is for playing/visualization (vs training).

  Returns:
    Environment configuration with electrical motors and metrics.
  """
  # Start with standard G1 flat config
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Load motor specs from database
  # G1 uses two motor types:
  # - Unitree 7520-14 for hip/knee (high torque)
  # - Unitree 5020-9 for ankle/arms (lower torque)
  hip_knee_motor = load_motor_spec("unitree_7520_14")  # 88 N⋅m continuous
  ankle_arm_motor = load_motor_spec("unitree_5020_9")  # 20 N⋅m continuous

  # Replace standard actuators with electrical motor actuators
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      # Hip and knee joints - high torque motors
      ElectricalMotorActuatorCfg(
        target_names_expr=(
          "left_hip_pitch",
          "right_hip_pitch",
          "left_hip_roll",
          "right_hip_roll",
          "left_hip_yaw",
          "right_hip_yaw",
          "left_knee",
          "right_knee",
        ),
        motor_spec=hip_knee_motor,
        stiffness=200.0,
        damping=10.0,
        saturation_effort=hip_knee_motor.peak_torque,
        velocity_limit=hip_knee_motor.no_load_speed,
        effort_limit=hip_knee_motor.continuous_torque,
      ),
      # Ankle and arm joints - lower torque motors
      ElectricalMotorActuatorCfg(
        target_names_expr=(
          "left_ankle_pitch",
          "right_ankle_pitch",
          "left_ankle_roll",
          "right_ankle_roll",
          "left_shoulder_pitch",
          "right_shoulder_pitch",
          "left_shoulder_roll",
          "right_shoulder_roll",
          "left_shoulder_yaw",
          "right_shoulder_yaw",
          "left_elbow_pitch",
          "right_elbow_pitch",
          "left_elbow_roll",
          "right_elbow_roll",
        ),
        motor_spec=ankle_arm_motor,
        stiffness=200.0,
        damping=10.0,
        saturation_effort=ankle_arm_motor.peak_torque,
        velocity_limit=ankle_arm_motor.no_load_speed,
        effort_limit=ankle_arm_motor.continuous_torque,
      ),
    )
  )

  # Add battery manager
  battery_spec = load_battery_spec("unitree_g1_9ah")  # 9Ah, 21.6V Li-ion
  cfg.scene.battery = BatteryManagerCfg(
    battery_spec=battery_spec,
    entity_names=("robot",),
    initial_soc=1.0,  # Start at 100%
    enable_voltage_feedback=True,  # Motor performance degrades as battery drains
  )

  # Add electrical metrics for real-time visualization
  # These will automatically appear in Viser viewer's Metrics tab
  cfg.metrics = {
    **(cfg.metrics or {}),  # Keep existing metrics
    **electrical_metrics_preset(),  # Add all 10 electrical metrics
  }

  # Optional: Add per-joint metrics for specific joints you want to monitor
  # Uncomment to track individual joint currents:
  # from mjlab.envs.mdp.metrics import motor_current_joint
  # from mjlab.managers import MetricsTermCfg
  # cfg.metrics.update({
  #     "left_knee_current": MetricsTermCfg(
  #         func=motor_current_joint,
  #         params={"joint_name": "left_knee"},
  #     ),
  #     "right_hip_current": MetricsTermCfg(
  #         func=motor_current_joint,
  #         params={"joint_name": "right_hip_pitch"},
  #     ),
  # })

  # Optional: Add cumulative energy tracking
  # Tracks total energy consumed over episode (resets on episode boundary)
  # Uncomment to enable:
  # from mjlab.envs.mdp.metrics import CumulativeEnergyMetric
  # cfg.metrics.update({
  #     "energy_consumed_wh": MetricsTermCfg(func=CumulativeEnergyMetric()),
  # })

  # Adjust action scale for electrical motors if needed
  # (May need tuning based on motor response characteristics)
  # cfg.actions["joint_pos"].scale = G1_ACTION_SCALE

  return cfg


def unitree_g1_rough_electric_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain config with electrical motors/battery.

  Same as flat config but for rough terrain.
  """
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

  # Start with rough terrain config
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Apply same electrical motor/battery setup as flat config
  # (Copy the motor actuator and battery setup from above)
  hip_knee_motor = load_motor_spec("unitree_7520_14")
  ankle_arm_motor = load_motor_spec("unitree_5020_9")

  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=(
      ElectricalMotorActuatorCfg(
        target_names_expr=(
          "left_hip_pitch",
          "right_hip_pitch",
          "left_hip_roll",
          "right_hip_roll",
          "left_hip_yaw",
          "right_hip_yaw",
          "left_knee",
          "right_knee",
        ),
        motor_spec=hip_knee_motor,
        stiffness=200.0,
        damping=10.0,
        saturation_effort=hip_knee_motor.peak_torque,
        velocity_limit=hip_knee_motor.no_load_speed,
        effort_limit=hip_knee_motor.continuous_torque,
      ),
      ElectricalMotorActuatorCfg(
        target_names_expr=(
          "left_ankle_pitch",
          "right_ankle_pitch",
          "left_ankle_roll",
          "right_ankle_roll",
          "left_shoulder_pitch",
          "right_shoulder_pitch",
          "left_shoulder_roll",
          "right_shoulder_roll",
          "left_shoulder_yaw",
          "right_shoulder_yaw",
          "left_elbow_pitch",
          "right_elbow_pitch",
          "left_elbow_roll",
          "right_elbow_roll",
        ),
        motor_spec=ankle_arm_motor,
        stiffness=200.0,
        damping=10.0,
        saturation_effort=ankle_arm_motor.peak_torque,
        velocity_limit=ankle_arm_motor.no_load_speed,
        effort_limit=ankle_arm_motor.continuous_torque,
      ),
    )
  )

  battery_spec = load_battery_spec("unitree_g1_9ah")
  cfg.scene.battery = BatteryManagerCfg(
    battery_spec=battery_spec,
    entity_names=("robot",),
    initial_soc=1.0,
    enable_voltage_feedback=True,
  )

  cfg.metrics = {
    **(cfg.metrics or {}),
    **electrical_metrics_preset(),
  }

  return cfg
