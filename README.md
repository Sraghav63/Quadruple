# Quadruple inverted pendulum

A realistic real-time simulation of a **quadruple inverted pendulum on a
servo-driven cart**, controlled the way real rigs are: a discrete LQR plus a
steady-state Kalman filter, designed from an identified model. The goal is to
build it for real (Teensy 4.1, 1 kHz loop), so the simulation models what the
hardware will do: encoder quantisation and noise, command delay, a
velocity-mode servo with limited bandwidth and force, joint stiction, and
random taps.

Details, conventions and the build/identification workflow are in
[`PROJECT_GUIDE.md`](PROJECT_GUIDE.md). Firmware bring-up is in
[`firmware/README.md`](firmware/README.md).

## Quick start

Python 3.11 or 3.12 (the dependency pins exclude 3.14).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Run the real-time viewer (pygame window, MuJoCo physics at wall-clock speed):

```bash
python scripts/run_realtime.py
```

| Input | Action |
| --- | --- |
| Left / Right, A / D | move the cart target 5 cm |
| Mouse drag in the scene | set the cart target |
| 1-4 | tap link 1-4 to the right (0.1 N for 20 ms); Shift reverses |
| Space | controller on / off |
| R | reset |
| P | pause |
| Esc | quit |

Options: `--speed` (0.1-2), `--seed`, `--tilt-deg` (initial random tilt),
`--no-controller`, `--headless SECONDS` (no window, prints a summary).

## Scripts

- `scripts/run_realtime.py`: interactive real-time viewer (above).
- `scripts/show3d.py`: simulate a scripted run and serve a browser 3D viewer
  from `viewer3d/` (`--seconds`, `--seed`, `--port`, `--no-open`, `--export-only`).
- `scripts/check_robustness.py`: seeded Monte Carlo of model mismatch, drive
  variation, delay and taps; reports the survival rate (`--trials`, `--workers`,
  `--min-success`, `--nominal`, per-effect `--*-scale` multipliers).
- `scripts/design_sweep.py`: screens link length, material and joint friction
  with the same disturbed trials (`--quick` for a fast pass).
- `scripts/export_firmware.py`: writes the gains and limits for the Teensy
  sketch (`--output`).

## Findings so far

- **Model-based, not RL.** Every verified real-hardware triple-pendulum success
  we found (TU Wien, Gluck/Eder/Kugi 2013; the Kaheman et al. benchmark rig)
  used model-based control: LQR or time-varying LQR with a Kalman filter, an
  identified model, and a velocity-mode servo cart. Real-hardware quadruple
  claims exist (Li Hongxing, about 2002, fuzzy control; Oh et al. 2025, RL) but
  are unverified and from single groups. The earlier SAC/PPO scaffold in this
  repo was dropped.
- **Speed matters.** The fastest unstable mode doubles in about 40 ms, so the
  loop needs roughly 500-1000 Hz. The old RL code ran at 50 Hz.
- **Joint stiction is the number one build risk.** Stiction relative to link
  weight produces a slow limit cycle: links wander and the cart hunts. Use
  low-friction bearings (open or lightly shielded, light oil) and links heavy
  enough to dominate it. The default is 12 mm steel rod, 0.35 m; a 0.30 m
  version was less robust in Monte Carlo (about 85% survival vs about 97%).
- **Noise versus stiction is a tuning trade-off.** High gains amplify encoder
  noise into actuator chatter; low gains let stiction wander grow.
- **Only light taps are survivable.** About 0.1 N for 20 ms on the top link,
  with a 30 m/s^2 cart. Starting requires releasing the chain nearly perfectly
  upright (a jig); swing-up is future work.
- **Robustness check.** With random +/-2% lengths, +/-10% masses, friction x0.5-2,
  drive bandwidth +/-40%, occasional extra delay, random light taps and a
  reference step, survival is about 95% or better.
- **Hardware plan (rough BOM about $650).** Brushless servo in velocity mode
  (moteus r4.11 or ClearPath) with a GT2 belt, 1.2-1.5 m MGN12 rail, AS5047P
  (14-bit) or MT6835 (21-bit) magnetic encoder per joint, wired through slip
  rings or thin cable (not wireless), Teensy 4.1.

## Layout

```
src/quad_pendulum/
  params.py      rig description (links, joints, cart, drive, sensors, timing)
  linear.py      analytic dynamics and the linearised design model
  controller.py  discrete LQR + Kalman filter, C header export
  plant.py       MuJoCo ground truth, servo, taps, sensors
  model.py       MuJoCo XML generated from RigParams
  sim.py         closed loop: sensors -> controller -> delay -> plant
scripts/         viewer, robustness, design sweep, firmware export
firmware/        Teensy 4.1 skeleton and bring-up notes
viewer3d/        browser 3D viewer
tests/           plant, model, controller and closed-loop tests
```
