"""Cartpole balance and swingup environment configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg

try:
  from mjlab.envs.mdp.actions import PidControlActionCfg  # type: ignore
except ImportError:
  PidControlActionCfg = None  # type: ignore
from mjlab.envs.mdp.metrics import electrical_metrics_preset
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_CARTPOLE_XML: Path = Path(__file__).parent / "cartpole.xml"
_CARTPOLE_ELECTRIC_XML: Path = Path(__file__).parent / "cartpole_electric.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))

# Entity.


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))


_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlMotorActuatorCfg(target_names_expr=("slider",)),),
)

_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},
)

_SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": math.pi},
  joint_vel={".*": 0.0},
)


def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_CARTPOLE_ARTICULATION,
    init_state=_SWINGUP_INIT if swing_up else _BALANCE_INIT,
  )


# Observations.


def pole_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """Cosine and sine of the pole hinge angle. Shape: [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


# Rewards.

# dm_control uses value_at_margin=0.1 by default.
_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """Gaussian sigmoid tolerance: 1 at x=0, value_at_margin=0.1 at |x|=margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _GAUSSIAN_SCALE
  return torch.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """Quadratic sigmoid tolerance: 1 at x=0, 0 at |x|>=margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _QUADRATIC_SCALE
  return torch.clamp(1 - scaled**2, min=0.0)


def cartpole_smooth_reward(
  env: ManagerBasedRlEnv,
  cart_cfg: SceneEntityCfg = _CART_CFG,
  hinge_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """dm_control smooth cartpole reward: upright * centered * small_control * small_vel.

  Args:
    env: The environment.
    cart_cfg: Entity config selecting the slider joint.
    hinge_cfg: Entity config selecting the hinge joint.
  """
  asset: Entity = env.scene[cart_cfg.name]

  # Pole angle cosine.
  hinge_angle = asset.data.joint_pos[:, hinge_cfg.joint_ids].squeeze(-1)
  pole_cos = torch.cos(hinge_angle)
  upright = (pole_cos + 1) / 2

  # Cart position.
  cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
  centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

  # Control effort (raw action from the policy).
  control = env.action_manager.action.squeeze(-1)
  small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

  # Pole angular velocity.
  hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
  small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

  return upright * centered * small_control * small_velocity


# Environment config.


def _make_env_cfg(swing_up: bool = False) -> ManagerBasedRlEnvCfg:
  cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
  hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))

  actor_terms = {
    "cart_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": cart_cfg},
    ),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin,
      params={"asset_cfg": hinge_cfg},
    ),
    "cart_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": cart_cfg},
    ),
    "pole_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": hinge_cfg},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider",),
      scale=1.0,
    ),
  }

  slider_range = (-0.1, 0.1) if not swing_up else (0.0, 0.0)
  events = {
    "reset_slider": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": slider_range,
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
      },
    ),
    "reset_hinge": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.034, 0.034),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",)),
      },
    ),
  }

  rewards = {
    "smooth_reward": RewardTermCfg(
      func=cartpole_smooth_reward,
      weight=1.0,
      params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"cartpole": _get_cartpole_cfg(swing_up=swing_up)},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="cartpole",
      body_name="cart",
      distance=4.0,
      elevation=-15.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
    ),
    decimation=5,
    episode_length_s=50.0,
  )


def cartpole_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=False)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def cartpole_swingup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=True)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


# RL config.


def cartpole_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="cartpole",
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )


def cartpole_balance_pid_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Cartpole balance environment with PID controller.

  This variant uses a classical PID controller instead of a learned policy.
  The controller reads observations and computes control forces automatically,
  making it work with dummy agents (--agent zero).

  Args:
    play: If True, configure for play mode (long episodes, no corruption).

  Returns:
    Environment configuration with PID control action.
  """
  if PidControlActionCfg is None:
    raise NotImplementedError(
      "PidControlActionCfg is not yet implemented. "
      "Use cartpole_balance_env_cfg() or cartpole_constant_rotation_env_cfg() instead."
    )

  cfg = _make_env_cfg(swing_up=False)

  # Replace effort action with PID control action
  cfg.actions = {
    "pid": PidControlActionCfg(  # type: ignore[misc]
      entity_name="cartpole",
      joint_names=("slider",),
      # PID gains (tuned for inverted pendulum balance)
      kp_angle=100.0,  # Strong angle correction (increased - most important!)
      kd_angle=20.0,  # Increased damping for stability
      kp_position=5.0,  # Gentle cart centering (reduced - secondary objective)
      kd_velocity=8.0,  # Moderate velocity damping
      ki_angle=0.0,  # No integral (disabled for stability)
      # Limits
      integral_limit=10.0,
      output_limit=100.0,
      # Targets
      target_angle=0.0,  # Upright
      target_position=0.0,  # Center
      # Observation keys
      angle_obs_name="pole_angle",
      angle_vel_obs_name="pole_vel",
      position_obs_name="cart_pos",
      velocity_obs_name="cart_vel",
    ),
  }

  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False

  return cfg


def cartpole_balance_electric_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Cartpole balance with electrical servo motor and real-time metrics.

  This variant uses an electrical motor actuator with realistic electrical and
  thermal dynamics, plus real-time visualization of power, current, and torque.

  Args:
    play: If True, configure for play mode (long episodes, no corruption).

  Returns:
    Environment configuration with electrical motor and metrics.
  """
  # Start from base configuration (not PID since it's not implemented yet)
  cfg = _make_env_cfg(swing_up=False)

  # Replace entity XML to use electrical motor version
  # Auto-discovery will automatically create ElectricalMotorActuator from XML
  def _get_electric_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_CARTPOLE_ELECTRIC_XML))

  cfg.scene.entities = {
    "cartpole": EntityCfg(
      spec_fn=_get_electric_spec,
      init_state=_BALANCE_INIT,
      # Let auto_discover_motors create the electrical actuator from XML
      auto_discover_motors=True,
    )
  }

  # Enable battery auto-discovery from XML
  cfg.scene.auto_battery = True

  # Add electrical metrics for real-time visualization
  cfg.metrics = {
    **electrical_metrics_preset(entity_name="cartpole"),  # Match entity name
  }

  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False

  return cfg


def cartpole_constant_rotation_env_cfg(
  motor_xml_path: str,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """CartPole with constant rotation for motor comparison.

  Applies sinusoidal torque commands to demonstrate motor characteristics.
  Does not use PID control - just direct torque commands.

  Args:
    motor_xml_path: Path to CartPole XML with specific motor configuration
    play: If True, disable corruption and extend episode length

  Returns:
    Environment configuration with constant torque action.
  """
  from mjlab.actuator import ElectricalMotorActuatorCfg
  from mjlab.motor_database import load_motor_spec

  cfg = _make_env_cfg(swing_up=False)  # Base configuration

  # Replace entity XML with custom motor variant
  def _get_custom_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(motor_xml_path)

  # Load motor spec from XML to manually configure with zero PD gains
  # Parse XML to find motor spec name
  spec = _get_custom_spec()
  motor_spec_name = None
  for custom_text in spec.texts:
    if custom_text.name == "motor_slide_motor":
      # Format is "motor_spec:motor_name"
      motor_spec_name = custom_text.data.split(":")[-1]
      break

  if motor_spec_name is None:
    raise ValueError(f"No motor_spec found in {motor_xml_path}")

  motor_spec = load_motor_spec(motor_spec_name)

  # Create electrical motor config with ZERO PD gains for direct torque control
  electrical_motor_cfg = ElectricalMotorActuatorCfg(
    target_names_expr=("slider",),
    motor_spec=motor_spec,
    saturation_effort=motor_spec.peak_torque,
    effort_limit=motor_spec.peak_torque,
    velocity_limit=motor_spec.no_load_speed,
    stiffness=0.0,  # ZERO PD gains - direct torque control only
    damping=0.0,
  )

  cfg.scene.entities = {
    "cartpole": EntityCfg(
      spec_fn=_get_custom_spec,
      init_state=_BALANCE_INIT,
      articulation=EntityArticulationInfoCfg(
        actuators=(electrical_motor_cfg,),
      ),
    )
  }

  # Enable battery tracking
  cfg.scene.auto_battery = True

  # Add electrical metrics
  cfg.metrics = {
    **electrical_metrics_preset(entity_name="cartpole"),
  }

  # Use direct torque control (not PID)
  cfg.actions = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider",),
      scale=1.0,
    ),
  }

  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False

  return cfg
