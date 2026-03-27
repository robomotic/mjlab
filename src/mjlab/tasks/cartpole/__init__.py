from mjlab.tasks.cartpole.cartpole_env_cfg import (
  cartpole_balance_electric_env_cfg,
  cartpole_balance_env_cfg,
  cartpole_constant_rotation_env_cfg,
  cartpole_ppo_runner_cfg,
  cartpole_swingup_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Cartpole-Balance",
  env_cfg=cartpole_balance_env_cfg(),
  play_env_cfg=cartpole_balance_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Cartpole-Swingup",
  env_cfg=cartpole_swingup_env_cfg(),
  play_env_cfg=cartpole_swingup_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)

# TODO: Re-enable when PidControlActionCfg is implemented
# register_mjlab_task(
#   task_id="Mjlab-Cartpole-Balance-PID",
#   env_cfg=cartpole_balance_pid_env_cfg(),
#   play_env_cfg=cartpole_balance_pid_env_cfg(play=True),
#   rl_cfg=cartpole_ppo_runner_cfg(),
# )

register_mjlab_task(
  task_id="Mjlab-Cartpole-Balance-Electric",
  env_cfg=cartpole_balance_electric_env_cfg(),
  play_env_cfg=cartpole_balance_electric_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Cartpole-Constant-Rotation",
  env_cfg=cartpole_constant_rotation_env_cfg(
    "src/mjlab/tasks/cartpole/cartpole_electric.xml",
    play=False,
  ),
  play_env_cfg=cartpole_constant_rotation_env_cfg(
    "src/mjlab/tasks/cartpole/cartpole_electric.xml",
    play=True,
  ),
  rl_cfg=cartpole_ppo_runner_cfg(),
)
