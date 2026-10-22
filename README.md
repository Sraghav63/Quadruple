# Quadruple inverted pendulum

A realistic real-time simulation of a **quadruple inverted pendulum on a
servo-driven cart**, controlled the way real rigs are: a discrete LQR plus a
steady-state Kalman filter, designed from an identified model. The goal is to
build it for real (Teensy 4.1, 1 kHz loop), so the simulation models what the
hardware will do: encoder quantisation and noise, command delay, a
velocity-mode servo with limited bandwidth and force, joint stiction, and
random taps. It also swings the chain between up/down configurations in real
time: it starts with every link hanging at rest, and asking for `UUUD` makes it
swing there while the loop keeps running.

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

Planning new transitions needs CasADi: `python -m pip install -e ".[plan]"`. Running
the stored ones does not.

Run the real-time viewer (pygame window, MuJoCo physics at wall-clock speed).
It starts with every link hanging down at rest:

```bash
python scripts/run_realtime.py
python scripts/run_realtime.py --goto UUUD     # swing there at the start
```

| Input | Action |
| --- | --- |
| Left / Right, A / D | move the cart target 5 cm |
| Mouse drag in the scene | set the cart target |
| 1-4 | tap link 1-4 to the right (0.1 N for 20 ms); Shift reverses |
| Space | controller on / off |
| 5-8 | toggle the target of link 1-4 between U and D |
| C / Shift+C | cycle the target through all configurations; U all up, H all down |
| Enter or G | go: swing to the chosen target while the loop keeps running |
| R | reset to all links down at rest |
| P | pause |
| Esc | quit |

The side panel shows the mode (hold, center, transition), the target, the
current edge with a progress bar, and the pending pick. The planned nominal
chain is drawn faintly during a swing. A new request is refused while a swing
is running.

Options: `--speed` (0.1-2), `--seed`, `--tilt-deg` (initial random tilt),
`--config UUUD` (starting configuration, default all down), `--goto CFG`
(request a configuration at the start), `--preset` (`quad`, `links3`, `links2`,
`links1`, `quad-short`), `--no-controller`, `--headless SECONDS` (no window,
prints a summary; with `--goto` it reports whether it arrived, how long it took
and the worst deviation from the nominal swing).

## Configurations

Any combination of links up or down can be balanced, not only all up. A
configuration is a string of `U` and `D`, one per link, **bottom link first**:
`UUUU` is the classic quadruple pendulum, `UUUD` has the top link hanging
down, `DDUU` has the two lower links down and the two upper links up. There
are 16 combinations (`quad_pendulum.params.all_configurations`). Each is an
equilibrium with its own linearisation and its own gains; link angles are
controlled as deviations from the target (0 for up, pi for down).

- `UUUU` keeps the original tuning. Any configuration with a `D` uses a
  stronger joint-bend penalty, a faster angle estimate and a weaker
  low-frequency attitude cost (see `controller.default_config`).
- `DDDD` is open-loop stable (every link hangs down); the controller then only
  damps it and holds the cart at the target.
- The sim starts in `--config` (default all down), tilted slightly about it.
  Moving to another configuration is a real-time transition, see below.
- Every configuration of the default rig survives a 20 cm reference step and
  0.08 N taps in MuJoCo without mismatch. Under the full Monte Carlo
  (`scripts/check_robustness.py --all-configs --trials 40`) 14 of 16 reach
  100%. `DUUU` (78%) and `DDUU` (75%) are the weak ones: three or two inverted
  links stacked on hanging ones, which the cart can only reach through the
  swinging links. Tuning sweeps plateaued at about 75-90% for both (it varies
  with the seeds); the upright tuning gave about 55% for `DDUU`.

## Transitions between configurations

Everything is real time: the loop never resets. Asking for a configuration
makes the chain swing there from whatever configuration it is in.

- **Offline planning.** `scripts/plan_transitions.py` finds each motion by
  direct collocation (CasADi + IPOPT) with limits: |x| <= 0.5 m, 2.2 m/s,
  25 m/s^2 command, 120 N at the belt, or a gentler 0.45 m, 1.8 m/s, 18 m/s^2,
  100 N when the fast plan does not track reliably. Each candidate is then
  checked closed loop in MuJoCo, starting after 6 s of balancing. The knots are stored as JSON in
  `data/transitions/<preset>/<START>-<GOAL>.json` with an `index.json`.
- **Run time.** `TransitionController` samples the knots at 1 kHz, linearises
  the chain along the nominal motion and designs a time-varying LQR and a
  time-varying Kalman filter in about 0.1 s (cached per edge). No optimisation
  runs in the loop.
- **Supervisor modes.** `hold`: the balancing controller of the current
  configuration. `center`: after a request, still balancing while the cart comes
  within 5 cm of the middle (the chain never fully settles because of joint
  stiction, so the swing starts from its normal wobble). `transition`: the tracker runs
  the planned swing. At the end the balancing controller of the new
  configuration takes over, seeded from the tracker's estimate.
- **Routing.** The library is sparse: swing-ups from all down, swing-downs to
  all down, and single-link flips. `TransitionLibrary.route` chains them
  (quickest by duration), so every pair is reachable, for example `UUUU` to
  `DDUU` goes through all down.
- **Fallen** during a transition means any link more than 0.6 rad from the
  nominal, or the rail end.

Library (default rig `quad`): 83 planned transitions; every pair of the 16
configurations is reachable, in 1.7 hops and 4.9 s on average (at most 3 hops,
8.5 s). `links3` has 32 (all pairs), `quad-short` 82 (all pairs), `links2` 10,
`links1` 2. Of the 204 three- and four-link edges tried, 197 passed validation;
the 7 that did not all leave a top-heavy configuration (for example
`UDUU` -> `UUUU`) and are routed around.

How reliable they are (`check_robustness.py --all-transitions --trials 8`, each
trial balancing 6 s first, then holding 3 s at the goal):

- With the controller's model equal to the plant (`--nominal`, what the
  simulator shows by default): 655/664 trials arrived (98.6%); 78 of the 83
  quad edges in every trial, the weakest `UUUU` -> `UUDU` at 5/8.
- With the full model mismatch of the balancing Monte Carlo (lengths +/-2%,
  masses +/-10%, friction x0.5-2, drive bandwidth +/-40%, sometimes an extra
  period of delay): 219/664 (33%). Link mass and length errors dominate; even
  masses within +/-1% and lengths within +/-0.2% leave `DDDD` -> `UUUU` at about
  half. Retuning the tracker's weights did not change this. Fast four-link swings
  are sensitive to the model, so a real build needs carefully identified masses,
  centres of mass and lengths (or online re-planning, not implemented). The
  website's "reality" knobs show this directly.

Regenerate or extend the library (resumable; existing files are kept unless
`--force`):

```bash
python -m pip install -e ".[plan]"
python scripts/plan_transitions.py                    # every preset
python scripts/plan_transitions.py --preset quad --workers 8
python scripts/plan_transitions.py --preset links2 --pairs DD-UU,UU-DD --force
```

Check the robustness of a transition (same mismatch, delay and bandwidth
perturbations as the balancing checks; the controller keeps the nominal rig):

```bash
python scripts/check_robustness.py --transition DDDD-UUUU --trials 30
python scripts/check_robustness.py --all-transitions --trials 20 --workers 8
```

The firmware does not run transitions yet (see `firmware/README.md`).

## Website

The browser version lives in [`web/`](web/README.md) (Vite + TypeScript, no
backend). From `web/`: `npm install`, then `npm run dev` for a local server,
`npm test` for the parity tests, and `npm run build` to write static files to
`web/dist/`. Deploy recipes for Netlify, Vercel, GitHub Pages and Cloudflare
Pages are in the web README.

## Scripts

- `scripts/run_realtime.py`: interactive real-time viewer (above).
- `scripts/plan_transitions.py`: plan and validate the transition library
  (`--preset`, `--pairs`, `--workers`, `--force`; needs the `plan` extra).
- `scripts/show3d.py`: simulate a run and serve a browser 3D viewer
  from `viewer3d/` (`--seconds`, `--seed`, `--config`, `--goto CFG` to show a
  swing from `--config`, default all down, `--preset`, `--port`, `--no-open`,
  `--export-only`).
- `scripts/check_robustness.py`: seeded Monte Carlo of model mismatch, drive
  variation, delay and taps; reports the survival rate (`--trials`, `--workers`,
  `--min-success`, `--nominal`, `--config CFG`, `--all-configs` (survival table
  per configuration), `--transition START-GOAL` (repeatable) and
  `--all-transitions` (per-edge table of success rate, median arrival time and
  worst deviation from the nominal), `--preset`, per-effect `--*-scale`
  multipliers).
- `scripts/design_sweep.py`: screens link length, material and joint friction
  with the same disturbed trials (`--quick` for a fast pass).
- `scripts/export_firmware.py`: writes the gains and limits for the Teensy
  sketch (`--output`, `--config`).

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
  at the target equilibrium, or by a planned transition from another one.
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
  sim.py         closed loop: sensors -> controller -> delay -> plant, supervisor modes
  transition.py  planned transitions: collocation, time-varying LQR + Kalman, library
  presets.py     named rigs (quad, links3, links2, links1, quad-short)
data/transitions/  stored transition knots per preset
scripts/         viewer, robustness, planner, design sweep, firmware export
firmware/        Teensy 4.1 skeleton and bring-up notes
viewer3d/        browser 3D viewer
tests/           plant, model, controller and closed-loop tests
```
