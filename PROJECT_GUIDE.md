# Quadruple Inverted Pendulum RL Guide

This project simulates a cart balancing a 1-, 2-, 3-, or 4-link inverted
pendulum using MuJoCo physics and Stable-Baselines3 reinforcement learning.

## What Is In This Repo

- `src/quad_pendulum/model.py`: generates the MuJoCo cart-pendulum XML model.
- `src/quad_pendulum/env.py`: Gymnasium environment used for training and rollout.
- `scripts/train.py`: trains SAC or PPO policies.
- `scripts/enjoy.py`: stable 2D pygame viewer with policy, mouse, and zero-action modes.
- `scripts/show3d.py`: exports a rollout and serves a browser-based Three.js 3D viewer.
- `scripts/record.py`: records an MP4 using offscreen MuJoCo rendering.
- `viewer3d/`: browser viewer files.
- `models/`: saved trained policies.
- `runs/`: TensorBoard/evaluation logs.

## Setup On CachyOS / Arch Linux

From the project folder:

```fish
cd /home/raghavp/Documents/Code/Quadruple
source .venv/bin/activate.fish
```

If you need to recreate the environment:

```fish
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate.fish
uv pip install -e ".[rl]"
```

Check the environment:

```fish
python scripts/train.py --links 1 --algo sac --timesteps 1000 --check-env
```

## Training

Training is headless. It does not open a simulation window. Use `enjoy.py`,
`show3d.py`, or `record.py` after training to view the result.

The environment includes realism constraints by default: one control-step
latency, first-order actuator lag, force slew limits, action deadband, motor
noise, small random cart disturbances, hinge friction, observation noise, and a
stricter fall threshold for upright balancing. These make the task harder and
invalidate older policies trained against the instantaneous-control environment.

Train one link:

```fish
python scripts/train.py --links 1 --algo sac --timesteps 200000 --overwrite
```

Train two links:

```fish
python scripts/train.py --links 2 --algo sac --timesteps 300000 --overwrite
```

Train three links:

```fish
python scripts/train.py --links 3 --algo sac --timesteps 600000 --overwrite
```

Train four links:

```fish
python scripts/train.py --links 4 --algo sac --timesteps 1000000 --overwrite
```

SAC is the recommended first algorithm. PPO is also available:

```fish
python scripts/train.py --links 2 --algo ppo --timesteps 500000 --overwrite
```

Saved models go to `models/`. For example, the two-link SAC model is:

```text
models/sac_links2.zip
```

## Running The 2-Link Version

Train it:

```fish
python scripts/train.py --links 2 --algo sac --timesteps 300000 --overwrite
```

Run the trained two-link policy in the 2D viewer:

```fish
python scripts/enjoy.py --links 2 --control policy --model models/sac_links2.zip --slowdown 2
```

The shorter version also works after training:

```fish
python scripts/enjoy.py --links 2 --slowdown 2
```

Run the two-link version in the 3D browser viewer:

```fish
python scripts/show3d.py --links 2 --model models/sac_links2.zip
```

Then open the URL printed in the terminal, usually:

```text
http://127.0.0.1:8765/
```

Stop the 3D server with `Ctrl+C`. If it was started in the background, stop it
with:

```fish
fuser -k 8765/tcp
```

## 2D Viewer

Run a trained policy:

```fish
python scripts/enjoy.py --links 1 --control policy --model models/sac_links1.zip --slowdown 2
```

Mouse-control mode:

```fish
python scripts/enjoy.py --links 1 --control mouse
```

In mouse mode:

- Drag inside the window to move the cart target.
- Press `R` to reset.
- Press `Esc` to quit.

Zero-action mode, useful for seeing the pendulum fall:

```fish
python scripts/enjoy.py --links 4 --control zero
```

## 3D Browser Viewer

The 3D viewer uses Three.js in your browser. It avoids MuJoCo's live GLFW viewer,
which can crash on Wayland/libdecor setups.

One-link 3D:

```fish
python scripts/show3d.py --links 1 --model models/sac_links1.zip
```

Two-link 3D:

```fish
python scripts/show3d.py --links 2 --model models/sac_links2.zip
```

Four-link 3D without a trained policy:

```fish
python scripts/show3d.py --links 4 --control zero
```

The script writes:

```text
viewer3d/trajectory.json
```

Then it starts a local server and opens the viewer. Drag in the browser to orbit
the camera.

## Recording A Video

Record one link:

```fish
python scripts/record.py --links 1 --model models/sac_links1.zip --output videos/links1.mp4
```

Record two links:

```fish
python scripts/record.py --links 2 --model models/sac_links2.zip --output videos/links2.mp4
```

Open the video:

```fish
xdg-open videos/links2.mp4
```

## TensorBoard

Training logs are written under `runs/`.

Start TensorBoard:

```fish
tensorboard --logdir runs
```

Open:

```text
http://localhost:6006
```

## Current Viewer Notes

- `enjoy.py` uses pygame by default and is the recommended live viewer.
- The original MuJoCo live viewer is still available:

```fish
python scripts/enjoy.py --links 1 --mujoco-viewer
```

On your Wayland setup, the MuJoCo GLFW viewer may show `libdecor` warnings or
segfault. Use the pygame viewer or 3D browser viewer instead.

## Troubleshooting

If `python -m pip` says pip is missing, use:

```fish
uv pip install -e ".[rl]"
```

If the cart barely moves or falls quickly, make sure you are using the latest
code and retrain. Older models may have been trained before the rail collision
fix or before actuator/sensor realism was enabled:

```fish
python scripts/train.py --links 1 --algo sac --timesteps 200000 --overwrite
```

If the trained model falls quickly, increase timesteps:

```fish
python scripts/train.py --links 2 --algo sac --timesteps 700000 --overwrite
```

If 3D viewer port `8765` is busy, pass another port:

```fish
python scripts/show3d.py --links 2 --model models/sac_links2.zip --port 8770
```

## Summer Build Plan

1. Get 1-link balancing reliably with the realism constraints enabled.
2. Train 2-link SAC until it balances from small random initial angles.
3. Train 3-link and 4-link versions.
4. Tune reward weights and reset noise.
5. Tune disturbance strength and curriculum settings.
6. Add a separate swing-up task later.

The current environment is an upright-balancing task. Swing-up is a harder
second phase because it needs different rewards and reset states.
