# Project guide

This project simulates, and is meant to guide the construction of, a quadruple
inverted pendulum on a belt-driven cart. Control is discrete LQR plus a
steady-state Kalman filter designed from a physical model of the rig. The
simulation is deliberately harsher than a textbook model so that a design that
works here has a fair chance on hardware. It is not a guarantee: nothing here
has been run on a real rig yet.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
```

Python 3.11 or 3.12 is required by the dependency pins. Run scripts as
`python scripts/<name>.py` from the repository root.

## Why model-based control

Every verified real-hardware triple-pendulum success we found (TU Wien,
Gluck/Eder/Kugi 2013; the Kaheman et al. benchmark rig) used model-based
control: LQR or time-varying LQR with a Kalman filter, an identified model, and
a velocity-mode servo cart. Real-hardware quadruple claims exist (Li Hongxing,
about 2002, fuzzy; Oh et al. 2025, RL) but are unverified and from single
groups. The earlier version of this repo trained SAC/PPO policies at 50 Hz; that
was removed. The fastest unstable mode of the default rig doubles in about
40 ms, which calls for a 500-1000 Hz loop, and the hardware route that has been
shown to work is model-based.

## Architecture

All parameters live in one frozen dataclass tree, `RigParams`. The plant, the
controller, the MuJoCo XML and the firmware export are all generated from it, so
changing the rig description changes everything consistently.

| Module | Role |
| --- | --- |
| `params.py` | `RigParams`: `LinkParams` (length, rod mass, top mass), `JointParams` (Coulomb, viscous), `CartParams`, `DriveParams`, `SensorParams`, `TimingParams`; the `Measurement` type; angle conversions. |
| `linear.py` | Analytic n-link dynamics (independent of MuJoCo), linearisation about any up/down equilibrium, open-loop unstable poles, and the continuous design model `state_space`. |
| `controller.py` | `ControllerConfig` and `LQRController`: discretisation, delay augmentation, LQR and Kalman gains, the per-period `step`, and `to_c_header` export. |
| `model.py` | Generates the MuJoCo XML from `RigParams`. |
| `plant.py` | `Plant`: ground-truth MuJoCo mechanics, velocity-mode servo, taps, sensor model. |
| `sim.py` | `Simulation`: the closed loop at the control rate. |

### Conventions

- Encoder joint angles `q_i` are relative: link `i` against link `i-1` (link 0
  is the cart). Zero is straight up; positive tilts toward +x.
- A configuration is a string of `U`/`D`, one per link, bottom link first
  (`UUUD`: top link down). `parse_configuration` validates it (None means all
  up), `all_configurations(n)` lists the 2^n combinations, and
  `equilibrium_angles` gives the target absolute angles (0 up, pi down).
  `wrap_angle` wraps to [-pi, pi).
- Absolute angles are cumulative sums, `theta_i = q_1 + ... + q_i`
  (`relative_to_absolute`, `absolute_to_relative`).
- The controller output is a cart acceleration command in m/s^2.
- The controller sees only a `Measurement`: quantised cart position and the
  relative encoder angles. It never sees velocities or ground truth;
  `Plant.state()` is for drawing and diagnostics only.

### Configurations

The controller, `Simulation` and the firmware export all take a
`configuration`. The linear model is built about that equilibrium
(`linear.linearized`, `state_space`, `unstable_poles` take it too), and the
angle states are deviations from the target, `wrap_angle(theta - theta_eq)`.
`LQRController.theta_eq` holds the target and `theta_estimate` still returns
absolute angles. In `Simulation`, `reset(random_tilt_deg=...)` tilts about the
equilibrium and a run counts as fallen when any link is more than 0.6 rad from
its target, or at the rail end.

- Tuning: `controller.default_config(configuration)`. All-up keeps the original
  tuning; anything with a `D` uses `joint_angle_tol=0.005` (stiffer joints keep
  the chain from folding at the hinges between up and down links),
  `link_accel_noise=2.5` (faster angle estimate) and `angle_integral_tol=0.003`.
  `DUUU` and `DDUU` remain the least robust, at about 75-90% in the Monte Carlo
  against 100% for the other 14.
- `DDDD` has no unstable poles: it is open-loop stable, so the controller adds
  damping and cart positioning only.
- Switching configuration needs a reset. Swing-up and transitions between
  equilibria are not implemented; the start must be released near the target
  pose.

### Plant

`Plant` runs MuJoCo (implicitfast integrator, tight Newton solver) at
`physics_dt` (default 0.2 ms), five substeps per 1 ms control period. Each link
is a uniform rod plus a point mass at its top (next joint's bearing, encoder,
magnet, fasteners). Joints have Coulomb (`frictionloss`) and viscous friction.
The cart slides with its own Coulomb and viscous friction and rail limits.

The servo is modelled as a velocity-mode drive: the commanded acceleration (clamped
to `accel_limit`) is integrated into a velocity setpoint (clamped to
`velocity_limit`); a PI velocity loop with bandwidth `velocity_bandwidth` turns
the velocity error into belt force (clamped to `force_limit`), with conditional
anti-windup. `Plant.push` applies a horizontal force at the moving top of a link
for a given duration. A run counts as fallen if any link deviates more than 0.6
rad from its target angle or the cart touches the end of the rail.

### Sensor model

Each period the plant reports the cart position (quantised to
`cart_resolution`, optional noise) and each relative joint angle with Gaussian
noise (`encoder_noise`) added before quantisation to `2*pi / 2^encoder_bits`.

### Simulation loop and timing

One `Simulation.step()` is one control period: sample sensors, run the
controller on the measurement and `x_ref`, push the command into a queue of
`compute_delay_steps` entries, apply the oldest command to the plant for one
period. This models sample, compute and bus latency as whole periods. The
simulation can take a separate `controller_rig` (what the controller believes)
from `rig` (the truth) to test model mismatch. `reset` starts at the target
equilibrium (optionally tilted about it) or at given angles and tells the controller the true starting pose, like a rig released
from a jig after one sensor read. `set_reference` clamps the target to 80% of the
rail half-length.

## How the controller is designed

1. **Linear model.** `linear.state_space` linearises the chain about the target equilibrium (upright by default)
   with cart acceleration as the input: state `[x, v, a, theta_1..n, omega_1..n]`.
   `a` is the actual cart acceleration and is a first-order lag (`model_lag`)
   behind the command, representing the velocity-loop dynamics.
2. **Discretisation.** Zero-order hold at the control period (`expm` on the
   augmented matrix).
3. **Delay augmentation.** The `compute_delay_steps` commands already in flight
   are appended to the state, `z = [x - x_ref, v, a, theta, omega, u_{k-d}..u_{k-1}]`,
   so both the regulator and the filter account for them exactly.
4. **LQR.** `Q` is diagonal from Bryson-style tolerances (1 / tolerance^2) on
   position, velocity, link angles and rates, plus an optional extra penalty on
   joint (relative) angles and rates. `R` comes from the command tolerance.
   Solved with the discrete algebraic Riccati equation.
5. **Kalman filter.** Measurements are cart position and the n absolute angles
   (summed encoder readings, so noise grows with link index). Measurement
   covariance comes from quantisation (`q^2/12`) plus encoder noise; process
   noise acts on cart acceleration and link angular acceleration. The
   steady-state gain is from the dual DARE.
6. **Runtime.** Each period: shift the position coordinate if the reference
   changed, correct `z += L (y - C z)`, `u = clamp(-K z)`, predict
   `z = A z + B u`. That is two small matrix products, and `to_c_header` exports
   `K, L, A, B, C` for the firmware. The controller reports
   `closed_loop_radius` and `observer_radius` (spectral radii) as sanity checks.

## Realism features and what each models

- Encoder quantisation and noise: `encoder_bits`, `encoder_noise`.
- Cart position quantisation: `cart_resolution`.
- Compute and bus delay: `compute_delay_steps` whole control periods.
- Servo dynamics: velocity setpoint integration, finite bandwidth PI loop, and
  acceleration, velocity and force limits with anti-windup.
- Joint Coulomb and viscous friction: the stiction behind the slow limit cycle
  described below.
- Cart (rail and belt) friction.
- Rail end-stops: touching the end counts as a failure.
- Model mismatch between truth and controller (`controller_rig`), used by the
  robustness script.
- Taps at any link, with force and duration.

Not modelled: belt compliance or backlash, encoder non-linearity, cable drag,
joint flex, motor thermal limits, and discrete-time effects inside the servo.

## Findings that shape the build

- **Stiction is the main risk.** Joint stiction relative to link weight causes a
  slow limit cycle: the links wander and the cart hunts. Use low-friction
  bearings (open or lightly shielded, light oil) and links heavy enough to
  dominate the friction. The default is 12 mm steel rod at 0.35 m. The 0.30 m
  version was less robust (about 85% survival in the Monte Carlo against about
  97%).
- **Noise versus wander.** Sensor noise times high gains gives actuator chatter.
  Lowering gains (larger tolerances, a more expensive command) cuts chatter but
  lets stiction wander grow. The `ControllerConfig` fields below are where this
  trade-off is made.
- **Taps.** A quadruple pendulum tolerates only light taps: about 0.1 N for 20 ms
  on the top link with a 30 m/s^2 cart.
- **Starting.** The controller only works near its target equilibrium. Start by
  releasing the chain from a jig essentially perfectly at that pose (upright for
  `UUUU`). Swing-up is not implemented.
- **Robustness.** `check_robustness.py` draws random +/-2% lengths, +/-10%
  masses, friction x0.5-2, drive bandwidth +/-40%, occasional extra delay,
  random light taps and a reference step. Survival is about 95% or better.
  Figures drift with controller tuning, so rerun it rather than trusting these
  numbers.

## Tools

```bash
python scripts/run_realtime.py                 # interactive viewer
python scripts/run_realtime.py --headless 10   # no window, prints a summary
python scripts/run_realtime.py --config UUUD   # top link hanging down
python scripts/check_robustness.py --trials 100 --workers 4
python scripts/check_robustness.py --nominal   # only taps, no mismatch
python scripts/check_robustness.py --all-configs --trials 20 --workers 8
python scripts/design_sweep.py --quick
python scripts/show3d.py                       # browser 3D viewer
python scripts/export_firmware.py --config UUUU
```

- `run_realtime.py`: pygame viewer. Controls are in `README.md`. Options:
  `--speed`, `--seed`, `--tilt-deg`, `--config`, `--no-controller`,
  `--headless SECONDS`. C / Shift+C cycles through the 16 configurations
  (rebuilds the sim and resets). Shows the rig, the configuration, link angle
  errors from the target and rolling plots; down links hang below the rail.
- `check_robustness.py`: seeded trials with random mismatch. `--trials`,
  `--seconds`, `--workers`, `--seed`, `--min-success` (exit threshold),
  `--nominal`, `--push-scale`, `--config CFG` (default all up), `--all-configs`
  (runs every configuration and prints a survival table), and per-effect multipliers `--length-scale`,
  `--mass-scale`, `--cart-mass-scale`, `--friction-scale`, `--viscous-scale`,
  `--bandwidth-scale`, `--delay-scale` that widen or narrow each uncertainty
  (for delay, the extra-step probability divided by 0.2).
- `design_sweep.py`: screens link length, material (aluminium tube and rod,
  steel rod) and joint friction with the same disturbed trials, also reporting
  the open-loop unstable poles. `--quick`, `--workers`, `--seeds`, `--seconds`.
- `show3d.py`: runs a scripted headless simulation (reference moves and taps),
  writes `viewer3d/trajectory.json` and serves `viewer3d/` on localhost
  (`--seconds`, `--seed`, `--config`, `--port`, `--no-open`, `--export-only`).
- `export_firmware.py`: writes the controller gains, parameters and limits for
  the Teensy sketch (`--output`, `--config`; the header gets `QP_CONFIGURATION`
  and `QP_THETA_EQ`). See `firmware/README.md`.

## Adapting it to your real build

The intended workflow:

1. Edit `params.py` (or build a `RigParams`) to match the planned build: link
   lengths, rod and top masses, joint friction, cart mass, rail length, drive
   limits, encoder bits, loop rate.
2. Run `design_sweep.py` and `check_robustness.py` on that description. Change
   the design until survival is acceptable.
3. Watch it in `run_realtime.py`. Tap it, step the reference, look for chatter
   and wander.
4. Build. Hardware plan (rough BOM about $650): brushless servo in velocity mode
   (moteus r4.11 or ClearPath) with a GT2 belt; 1.2-1.5 m MGN12 rail; one
   AS5047P (14-bit) or MT6835 (21-bit) magnetic encoder per joint; wires via
   slip rings or thin cable, not wireless; Teensy 4.1 at 1 kHz.
5. Identify the real parameters: masses and centres of mass (include magnets,
   bearings, wiring), joint friction from swing-down tests, drive bandwidth,
   delay and limits from a velocity step test. `firmware/README.md` has the
   detailed procedure.
6. Update `params.py` with the measurements and rerun the checks.
7. Run `export_firmware.py` and follow the bring-up steps in
   `firmware/README.md` (jig in the target configuration, calibration, conservative limits, E-stop).
   The Teensy sketch is a skeleton: the drive and cart-position stubs fail
   closed and must be implemented.

Link geometry matters more than any tuning knob. Prefer materials and lengths
the design sweep says are robust, and treat bearing friction as part of the
design.

## Tuning knobs (`ControllerConfig`)

LQR weights are Bryson-style: each is the deviation you would accept, and the
cost is 1 / deviation^2. Smaller tolerance = heavier penalty.

- `cart_position_tol`: how far the cart may sit from the reference. Smaller pulls
  the cart back to the target harder, at the cost of the angle loops.
- `cart_velocity_tol`: penalty on cart speed; damps rail travel.
- `angle_tol`: penalty on each link's tilt from its target equilibrium.
- `rate_tol`: penalty on each link's angular rate; more damping.
- `accel_tol`: penalty on the command. Larger tolerance = cheaper command = more
  aggressive and noisier output. Smaller = quieter actuator, weaker wander
  rejection. The main chatter knob.
- `joint_angle_tol`: extra penalty on the bend at each joint (relative angles).
  Keeps the chain straight so a tap on the top link is not absorbed by whipping
  link 1 far outside the small-angle region. `None` turns it off.
- `joint_rate_tol`: same for relative joint rates; off by default because it
  increases chatter.
- `cart_accel_noise`: Kalman process noise on unmodelled cart acceleration.
  Larger trusts measurements more and the model less.
- `link_accel_noise`: process noise on link angular acceleration (friction,
  model error, taps). Larger gives a faster, noisier estimate.
- `extra`: free-form dict, unused by the design itself.

Rig-side knobs that interact with tuning: `compute_delay_steps` (must match the
real latency), `drive.model_lag` (about 1 / `velocity_bandwidth`),
`drive.accel_limit`, and the sensor noise values (they set the Kalman
measurement covariance). Keep the controller's `RigParams` honest; a design
that only works because the model has less noise or delay than the hardware
will not transfer.

## Testing

```bash
python -m pytest
```

- `test_model.py`: generated MuJoCo masses, inertias and centres of mass; the
  analytic dynamics against an independent model.
- `test_plant.py`: reset and angle conventions, exact control-period stepping,
  servo tracking, limits and anti-windup, quantisation and seeded noise, taps,
  free fall of a tilted chain, rail-end latch.
- `test_controller.py`: the default rig is open-loop unstable with four unstable
  modes, LQR and observer are stable, a one-link case, accel limit, header export.
- `test_closed_loop.py`: five seconds upright, light taps on every link, a 20 cm
  reference step, and falling with the controller disabled.

The Teensy firmware is not built or run by any Python test.

## Roadmap

- Joint friction identification and compensation, since stiction is the dominant
  failure mode.
- Swing-up and transitions between configurations (needed to move from one
  equilibrium to another without resetting).
- Swing-up via trajectory optimisation plus time-varying LQR, to remove the
  need for a jig.
- Optional RL layered on top of the model-based controller (for example residual
  or recovery policies), not replacing it.
- MPC with explicit input and rail constraints.
- Hardware bring-up and validation of the firmware.
