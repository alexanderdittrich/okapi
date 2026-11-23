# Showcase tasks

| Task | Files | Algorithm |
|---|---|---|
| Okapi joystick locomotion | [`okapi_locomotion/`](okapi_locomotion) | PPO |
| Penalty shoot-out (striker, then keeper) | [`penalties/`](penalties) | PPO, two-stage curriculum |
| Foot tricks (humanoid) | [`foot_tricks.ipynb`](foot_tricks.ipynb) | PPO |
| Head tricks (humanoid) | [`head_tricks.ipynb`](head_tricks.ipynb) | PPO |
| Robot tricks (OP3) | [`robot_tricks.ipynb`](robot_tricks.ipynb) | PPO |

```bash
uv run examples/okapi_locomotion/train.py   # trains, then writes docs/_static/okapi_locomotion.gif
uv run examples/penalties/train.py          # striker → keeper → docs/_static/penalties.gif
```

The football tasks are adapted from [goncalog/ai-robotics](https://github.com/goncalog/ai-robotics),
which does not state a license or sources.
