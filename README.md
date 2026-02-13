# mjlab example: anymal_c_velocity

![ANYmal C locomotion](teaser.gif)

Shows how to integrate a **custom robot** with an existing mjlab task. This
trains an ANYmal C quadruped to walk and track commanded body velocities.

## Structure

```
src/anymal_c_velocity/
  __init__.py                        # Task registration (entry point)
  env_cfgs.py                        # Environment configs (sensors, rewards, terminations)
  rl_cfg.py                          # RL hyperparameters (PPO)
  anymal_c/
    anymal_c_constants.py            # Robot definition (actuators, collision, init state)
    xmls/
      anymal_c.xml                   # MuJoCo MJCF model
      assets/                        # Meshes and textures
```

## How it works

### 1. Depend on mjlab

In `pyproject.toml`, depend on `mjlab` and declare an entry point so mjlab
auto-discovers your tasks on import:

```toml
[build-system]
requires = ["uv_build>=0.8.18,<0.9.0"]
build-backend = "uv_build"

[project]
dependencies = ["mjlab>=1.1.0"]

[project.entry-points."mjlab.tasks"]
anymal_c_velocity = "anymal_c_velocity"
```

The `[build-system]` table is required. Without a build backend, the package
won't be installed into the environment and the entry point won't be registered.

### 2. Define your robot

`anymal_c_constants.py` provides the robot's `EntityCfg`: a MuJoCo spec
(loaded from XML), actuator parameters, collision properties, and initial
joint state. This is the only part that is specific to your robot's hardware.

Key pieces:
- **`get_spec()`**: loads the MJCF XML and its mesh assets.
- **`BuiltinPositionActuatorCfg`**: PD gains, effort limits, armature.
- **`EntityCfg`**: ties together the spec, actuators, collisions, and init state.

### 3. Configure the environment

`env_cfgs.py` starts from `make_velocity_env_cfg()` (the built-in velocity
task defaults) and customizes it for your robot:

- Set `cfg.scene.entities` to your robot.
- Configure contact sensors (which geoms are feet, what counts as illegal contact).
- Tune reward weights, termination conditions, and viewer settings.
- Optionally set up terrain curriculum.

A `play=True` variant disables randomization for evaluation.

### 4. Configure RL

`rl_cfg.py` returns a `RslRlOnPolicyRunnerCfg` with PPO hyperparameters
(network architecture, learning rate, clip param, etc). Start with defaults
and tune from there.

### 5. Register tasks

`__init__.py` calls `register_mjlab_task()` for each variant (rough/flat),
passing the env config, RL config, and runner class:

```python
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Anymal-C",
  env_cfg=anymal_c_flat_env_cfg(),
  play_env_cfg=anymal_c_flat_env_cfg(play=True),
  rl_cfg=anymal_c_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

## Usage

```sh
# Sanity check: watch your robot stand and fall under zero actions
uv run play Mjlab-Velocity-Flat-Anymal-C --agent zero

# Train
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-Anymal-C \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3_000

# Play the trained checkpoint
uv run play Mjlab-Velocity-Flat-Anymal-C --wandb-run-path <wandb-run-path>
```
