# Showcases

Tasks beyond the benchmarks, in `examples/`. They show how to write custom
Playground environments and train them with okapi.

## Okapi locomotion

```{image} _static/okapi_locomotion.gif
:width: 560px
```

`examples/okapi_locomotion/`: joystick velocity tracking for a 12-DoF quadruped
okapi, trained with PPO (75M steps).

```bash
uv run examples/okapi_locomotion/train.py
```

Smooth gaits come from a **trot clock**. The gait phase is part of the observation, and
a `feet_phase` reward asks diagonal leg pairs to follow a sinusoidal swing-height
profile. Penalties on action rate, action jerk (second difference), vertical bounce and
roll/pitch rate remove the remaining jitter. The critic sees privileged, noise-free state.

The rainforest (`scenery.py`) and the okapi's coat are decoration only: no mass, no
collision. Training uses `OkapiJoystick(scenery=False)` because the extra geoms only cost time.

## Penalty shoot-out

```{image} _static/penalties.gif
:width: 560px
```

`examples/penalties/`: two torque-controlled humanoids, trained as a curriculum.

```bash
uv run examples/penalties/train.py        # striker, then keeper, then the GIF
```

1. **Striker** (20M steps; scores 79 % of shots against a passive keeper). Each episode
   samples a target point in the goal, which is part of the striker's observation. The rewards: approach the ball, ball velocity towards the
   target, a goal bonus plus an accuracy bonus near the target, and staying upright
   (falling ends the episode). The keeper holds its stance.
2. **Keeper** (45M steps; saves 74 % of shots, so the striker's rate falls from 78 % to
   26 %). The trained striker is frozen into the environment
   (`okapi.load_policy(...)` returns a jit-able function) and shoots at random targets
   that the keeper cannot see. The keeper is rewarded for standing ready before the
   kick, for moving its hands to the ball's predicted (ballistic) crossing point once
   the ball is struck, and for saves. Conceded goals are penalized. The keeper may dive.

The environment keeps the shot target and kick flags in `state.info`, so it trains with
`full_reset=True`: on episode ends the auto-reset calls `env.reset` instead of restoring a
cached first state.

## Football tricks

`examples/foot_tricks.ipynb`, `head_tricks.ipynb` and `robot_tricks.ipynb` keep a ball
in the air with a humanoid's foot, a humanoid's head and the OP3 robot's foot.

```{image} _static/foot_tricks_2.gif
:width: 420px
```

The football tasks are adapted from [goncalog/ai-robotics](https://github.com/goncalog/ai-robotics).
