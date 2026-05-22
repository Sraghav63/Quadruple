# Quadruple Inverted Pendulum RL

MuJoCo + Gymnasium + Stable-Baselines3 scaffold for training a cart to balance a
1- to 4-link inverted pendulum. The environment is written so you can start with
one link, then increase difficulty until the quadruple pendulum works.

For the full setup, training, 2D viewer, 3D viewer, recording, and troubleshooting
guide, see [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## Recommended Setup

Use Python 3.11 or 3.12 for this project. The local system Python may be newer
than the RL stack currently supports.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[rl]"
```

If you only have Python 3.14 installed, install Python 3.12 first and create the
virtual environment with that interpreter.

## Train

Start with the single pendulum:

```bash
python scripts/train.py --links 1 --algo sac --timesteps 100000 --overwrite
```

Then increase difficulty:

```bash
python scripts/train.py --links 2 --algo sac --timesteps 300000 --overwrite
python scripts/train.py --links 3 --algo sac --timesteps 600000 --overwrite
python scripts/train.py --links 4 --algo sac --timesteps 1000000 --overwrite
```

PPO is also supported:

```bash
python scripts/train.py --links 4 --algo ppo --timesteps 1000000
```

Training logs are written to `runs/`, and saved policies are written to
`models/`. If a model with the same name already exists, the training script
saves a numbered copy such as `models/sac_links1_run2.zip`. Add `--overwrite`
when you intentionally want to replace the default model file.

## Watch A Policy

```bash
python scripts/enjoy.py --links 1
```

Training itself is headless, so the window opens only when you run `enjoy.py`.
The viewer uses pygame by default so it avoids MuJoCo's Wayland/libdecor GLFW
crash path. For a slower replay:

```bash
python scripts/enjoy.py --links 1 --slowdown 2
```

To control the cart target with your mouse:

```bash
python scripts/enjoy.py --links 1 --control mouse
```

Drag in the window to move the target. Press `R` to reset and `Esc` to quit.

To view the raw simulation without a trained policy:

```bash
python scripts/enjoy.py --links 4 --control zero
```

To use the original MuJoCo live viewer anyway:

```bash
python scripts/enjoy.py --links 1 --mujoco-viewer
```

On Wayland systems, MuJoCo's live GLFW viewer can crash because of GLFW/libdecor
issues. If that happens, keep using the default pygame viewer or record an MP4:

```bash
python scripts/record.py --links 1 --model models/sac_links1.zip --output videos/links1.mp4
```

Then open `videos/links1.mp4` in your normal video player.

## 3D Browser Viewer

Generate a rollout from the trained model and view it in a Three.js browser scene:

```bash
python scripts/show3d.py --links 1 --model models/sac_links1.zip
```

The script writes `viewer3d/trajectory.json`, starts a local server, and opens
the viewer. Drag in the browser to orbit the camera. Stop the server with
`Ctrl+C` in the terminal.

## Project Direction

Good summer milestones:

1. Balance 1 link from small random initial angles.
2. Balance 2 links from small random initial angles.
3. Balance 4 links from small random initial angles.
4. Add stronger randomization and disturbances.
5. Add curriculum learning inside the 4-link environment by slowly increasing
   reset noise and disturbance strength.
6. Try swing-up from hanging or mixed initial states.

The current environment is intentionally focused on upright balancing. Swing-up
is a harder second phase because the reward and initial-state distribution need
to change. The 1-, 2-, and 3-link environments are practice stages; their neural
networks cannot be loaded directly into the 4-link environment because the
observation dimensions are different.
