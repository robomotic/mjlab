=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Added
^^^^^

- Extended ``MotorSpecification`` with 6 new optional fields from Maxon/Faulhaber
  datasheets: ``number_of_pole_pairs`` (int, for commutation frequency),
  ``commutation`` (str, sensor type), ``max_speed`` (float, bearing limit),
  ``weight`` (float, kg), ``friction_static`` (float, N⋅m), and ``friction_dynamic``
  (float, N⋅m⋅s/rad). All fields are optional with sensible defaults and not currently
  used in physics equations (reserved for future enhancements). No breaking changes
  to existing motor specifications.
- **PID-controlled cartpole**: Added ``Mjlab-Cartpole-Balance-PID`` task demonstrating
  classical PID control integration with mjlab. Includes standalone ``PidControlAction``
  action term that computes control forces from observations using tunable PID gains.
  Works with ``--agent zero`` for visualizing hand-tuned control policies without training.
- **Electrical cartpole with real-time metrics**: Added ``Mjlab-Cartpole-Balance-Electric``
  task demonstrating electrical motor actuator with battery visualization. Features custom
  servo motor specification (50 Nm stall torque, 48V, 0.5Ω resistance) and real-time
  electrical metrics (motor current, voltage, power, temperature, back-EMF, battery SOC,
  battery voltage, current, power, temperature). Uses PID control with electrical motor
  physics and battery dynamics. Motor and battery automatically discovered from XML via
  ``auto_discover_motors`` and ``auto_battery`` flags. Run with: ``uv run play
  Mjlab-Cartpole-Balance-Electric --agent zero``.
- **Motor database with electrical characteristics** (Phase 1): Added comprehensive
  motor database infrastructure for realistic electrical and thermal motor modeling.
  Includes ``MotorSpecification`` dataclass with electrical properties (resistance,
  inductance, motor constants), mechanical properties (gear ratio, inertia, torque limits),
  and thermal properties (thermal resistance, time constant, temperature limits).
  Database loader supports flexible path resolution (built-in, user directory, project
  directory, environment variable, URLs). XML integration enables storing motor specs
  in MuJoCo XML files via ``<custom><text>`` elements for sharing via MuJoCo Menagerie.
  Added ``load_motor_spec()``, ``write_motor_spec_to_xml()``, ``parse_motor_specs_from_xml()``
  functions. Includes 3 example motor specs (Unitree 7520-14, 5020-9, test motor).
  New files: ``motor_database/motor_spec.py``, ``motor_database/database.py``,
  ``motor_database/xml_integration.py``, ``motor_database/motors/*.json``.
  Tests: ``test_motor_database.py`` (18 tests), ``test_motor_xml.py`` (14 tests).
- **Electrical motor actuator** (Phase 2): Added ``ElectricalMotorActuator`` with full
  RL circuit electrical dynamics and first-order thermal modeling. Implements semi-implicit
  integration for numerical stability (V = I·R + L·dI/dt + Ke·ω). Tracks per-environment
  per-joint electrical state: current, voltage, back-EMF, power dissipation, winding
  temperature. Supports voltage clamping, temperature limits, and motor constant
  validation. Fully GPU-batched for efficient parallel simulation. Added
  ``ElectricalMotorActuatorCfg`` configuration class. New file:
  ``actuator/electrical_motor_actuator.py``. Tests: ``test_electrical_motor_actuator.py``
  (13 tests covering RL circuit dynamics, thermal dynamics, PD integration, batching).
- **Battery system and power management** (Phase 2B): Added battery database and
  ``BatteryManager`` for scene-level power management. Implements realistic voltage drop
  physics (V_terminal = V_oc(SOC) - I·R(SOC,T)), state-of-charge tracking (dSOC/dt =
  -I/(Q·3600)), and thermal dynamics (dT/dt = (I²R - (T-T_amb)/R_th)/τ_th). Supports
  multiple chemistries (LiPo, LiFePO4, Li-ion) with non-linear OCV curves and
  SOC-dependent internal resistance. Includes inverter efficiency modeling for PMSM
  motors (DC-to-AC conversion losses). Battery voltage dynamically limits motor
  performance via ``enable_voltage_feedback``. Aggregates current from all electrical
  motor actuators per step. Added ``BatterySpecification``, ``BatteryManager``,
  ``BatteryManagerCfg``, ``InverterCfg`` classes. New files: ``battery_database/
  battery_spec.py``, ``battery_database/database.py``, ``battery_database/xml_integration.py``,
  ``battery_database/batteries/*.json``, ``battery/battery_manager.py``,
  ``actuator/inverter.py``. Modified: ``scene/scene.py`` (battery integration hooks),
  ``actuator/electrical_motor_actuator.py`` (inverter support). Tests:
  ``test_battery_database.py`` (19 tests), ``test_battery_manager.py`` (24 tests).
- **Automatic motor/battery integration** (Phase 3): Added automatic discovery and
  integration of motor specs and battery from XML ``<custom><text>`` elements. Eliminates
  manual configuration - motors and battery are automatically created from XML annotations.
  Added ``auto_discover_motors`` flag to ``EntityCfg`` (default True) and ``auto_battery``
  flag to ``SceneCfg`` (default True). Manual configuration takes precedence over
  auto-discovery. Modified: ``entity/entity.py`` (``_auto_discover_motors()`` method),
  ``scene/scene.py`` (``_auto_discover_battery()`` method). Tests: ``test_auto_discovery.py``
  (7 tests). All 770 existing tests continue to pass (backward compatible).
- **Electrical validation and documentation** (Phase 4): Added performance benchmarks,
  datasheet validation tests, and energy conservation validation. Performance: ~63%
  overhead vs DC motor (acceptable for full RL+thermal physics), <100ms per step with
  1000 environments. Datasheet validation confirms Unitree 7520-14/5020-9 specs
  (stall torque, no-load speed, torque-speed curves, thermal limits). Energy conservation
  tests verify power balance and heat dissipation. Added comprehensive
  ``ElectricalMotorActuator`` documentation to actuators.rst with physics equations,
  usage examples, and validation references. New files: ``test_electrical_performance.py``
  (3 tests), ``validation/test_motor_datasheet_validation.py`` (8 tests),
  ``validation/test_energy_conservation.py`` (5 tests). Modified: ``docs/source/actuators.rst``.
- **Real-time motor/battery metrics visualization** (Phase 6): Added 15 electrical
  metric functions for real-time visualization through the existing Viser viewer
  infrastructure. Includes 5 aggregate motor metrics (``motor_current_avg``,
  ``motor_voltage_avg``, ``motor_power_total``, ``motor_temperature_max``,
  ``motor_back_emf_avg``), 5 battery metrics (``battery_soc``, ``battery_voltage``,
  ``battery_current``, ``battery_power``, ``battery_temperature``), 5 per-joint metrics
  (``motor_current_joint``, ``motor_voltage_joint``, ``motor_power_joint``,
  ``motor_temperature_joint``, ``motor_back_emf_joint``), and 2 cumulative metrics
  (``CumulativeEnergyMetric``, ``CumulativeMechanicalWorkMetric``). Added
  ``electrical_metrics_preset()`` helper for one-line setup. Metrics automatically
  appear in Viser viewer's Metrics tab with real-time plotting (300-point history,
  60 fps), checkbox filtering, and text search. Correctly visualizes regenerative
  braking (negative battery power when motors are backdriven by gravity). <2%
  performance overhead. Added ``Mjlab-Velocity-Flat-Unitree-G1-Electric`` and
  ``Mjlab-Velocity-Rough-Unitree-G1-Electric`` tasks demonstrating electrical motors
  with battery visualization. New/modified files: ``envs/mdp/metrics.py`` (+200 lines),
  ``tasks/velocity/config/g1/env_cfgs_electric.py``, ``tasks/velocity/config/g1/
  README_ELECTRIC.md``, ``tasks/velocity/config/g1/__init__.py``, ``examples/
  electrical_metrics_viz_simple.py``. Tests: ``test_electrical_metrics_advanced.py``
  (18 tests). Run with: ``uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric
  --agent zero --viewer viser``.
- **Cable-powered electrical motors**: Electrical motors can now operate without
  a battery for infinite power scenarios. Simply omit the ``battery`` configuration
  to enable cable-powered mode with no voltage sag or SOC depletion. Added
  ``Mjlab-Velocity-Flat-Unitree-G1-Electric-Cable`` task demonstrating cable power.
  The ``electrical_metrics_preset()`` helper now supports ``include_battery=False``
  for motor-only metrics. Use for benchtop testing, training without power constraints,
  or maximum performance evaluation. New/modified files: ``tasks/velocity/config/g1/
  env_cfgs_electric.py`` (+60 lines), ``tasks/velocity/config/g1/__init__.py`` (+8 lines),
  ``tasks/velocity/config/g1/README_ELECTRIC.md`` (+80 lines), ``docs/motors/
  design-proposal.md`` (+60 lines). Tests: ``test_cable_powered.py`` (3 tests).
  Run with: ``uv run play Mjlab-Velocity-Flat-Unitree-G1-Electric-Cable
  --agent zero --viewer viser``.
- **Regenerative braking control**: Added ``allow_regenerative_braking`` flag to
  ``BatteryManagerCfg`` to control whether batteries accept negative current from
  backdriven motors. Default is ``False`` (reject regenerative braking) for
  realistic simulation of commercial Li-Po/Li-ion batteries without charge circuits.
  Set to ``True`` only for future battery specs that explicitly support regenerative
  charging. When disabled, negative motor current is clamped to zero and energy
  dissipates as heat in motor windings. New/modified files: ``battery/battery_manager.py``
  (+18 lines docstring, +4 lines code), ``tasks/velocity/config/g1/README_ELECTRIC.md``
  (+40 lines). Tests: ``test_battery_manager.py`` (+3 regenerative braking tests).
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- Added ``DelayedBuiltinActuatorGroup`` that fuses delayed builtin actuators
  sharing the same delay configuration into a single buffer operation.
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).

Changed
^^^^^^^

- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).

Fixed
^^^^^

- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
