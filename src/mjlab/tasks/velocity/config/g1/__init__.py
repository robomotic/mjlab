from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .env_cfgs_electric import (
  unitree_g1_flat_electric_cable_env_cfg,
  unitree_g1_flat_electric_env_cfg,
  unitree_g1_rough_electric_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# New: G1 with electrical motors and battery metrics
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Electric",
  env_cfg=unitree_g1_flat_electric_env_cfg(),
  play_env_cfg=unitree_g1_flat_electric_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1-Electric",
  env_cfg=unitree_g1_rough_electric_env_cfg(),
  play_env_cfg=unitree_g1_rough_electric_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Cable-powered (infinite power) - electrical motors with NO battery
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Electric-Cable",
  env_cfg=unitree_g1_flat_electric_cable_env_cfg(),
  play_env_cfg=unitree_g1_flat_electric_cable_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
