# Teensy 4.1 controller skeleton

Generate gains from the repository root:

```sh
.venv/bin/python scripts/export_firmware.py                 # all links up
.venv/bin/python scripts/export_firmware.py --config UUUD   # top link down
```

The configuration is a string of `U`/`D`, one per link, bottom link first. It
is baked into `qp_gains.h` as `QP_CONFIGURATION` and `QP_THETA_EQ[]` (0 for up,
pi for down); the sketch controls `wrap(theta_i - QP_THETA_EQ[i])`. One build
balances one configuration. The shipped header is all up.

Open `teensy/quad_pendulum/quad_pendulum.ino` in Teensyduino, select Teensy 4.1,
and build with its standard Arduino, SPI and IntervalTimer libraries. No moteus
library is required until the two hardware stubs are implemented. Both stubs
fail closed: the sketch cannot arm as shipped. `qp_gains.h` records every design
parameter, limits, timestep and modeled command delay. The exporter repairs
integer float literals from `to_c_header()` (e.g. `0f` becomes `0.0f`) without
changing coefficients. Regenerate after any parameter/controller change.

## Wiring

| Signal | Teensy pin / connection |
| --- | --- |
| AS5047P SCK / MOSI / MISO | 13 / 11 / 12, shared SPI bus |
| Joint 1 / 2 / 3 / 4 CS | 10 / 9 / 8 / 7, independent active-low selects |
| Upright calibration button | 2 to ground, internal pull-up |
| Serial commands and telemetry | USB Serial, 115200 requested baud |
| Cart encoder | Implement `readCartPositionMeters`; home to rail centre |
| moteus CAN-FD | CAN3 TX 31 / RX 30 through a **CAN-FD transceiver**, then CANH/CANL |

Use a common signal ground, short SPI wiring and sensor-board decoupling. Teensy
I/O is 3.3 V and is not 5 V tolerant. Use an AS5047P board configured for 3.3 V
logic per its schematic/datasheet. Pull each CS high during reset. Terminate CAN
at the two bus ends with 120 ohm resistors. The Teensy pins cannot drive CANH/CANL
directly. Fit a hard E-stop that disables drive torque independently of software,
rail limit switches and mechanical catches for the links.

The encoder implementation uses SPI mode 1 at 1 MHz, pipelined ANGLECOM reads,
16-bit even parity and the response error flag. Each relative reading is wrapped
to [-pi, pi); absolute link tilts are cumulative sums. Positive relative angle
must mean rotation toward +x relative to the parent: verify and adjust each
`kEncoderSign`. Offsets are RAM-only and must be calibrated after every reset.

References: [AS5047P datasheet](https://look.ams-osram.com/m/d05ee39221f9857/original/AS5047P-DS000324.pdf),
[Teensy 4.1 pinout](https://www.pjrc.com/store/teensy41.html),
[IntervalTimer ISR constraints](https://www.pjrc.com/teensy/td_timing_IntervalTimer.html).

## Identify the rig before powered balance

1. Measure hinge-to-hinge lengths, rod and joint/top masses, and each link's COM.
   Include magnets, bearings, fasteners and wiring. Update `params.py` so its
   uniform-rod-plus-top-mass representation matches measurements; flag any COM or
   inertia that this representation cannot reproduce.
2. Swing-test each link separately, safely hanging downward. Fit oscillation
   period/inertia and decay versus amplitude/speed to identify Coulomb and viscous
   friction. Include bearing preload, seals, cabling and motor-off rail friction.
3. With links restrained, step-test the drive velocity response at bounded speed
   and force. Measure bandwidth (rad/s), acceleration lag, encoder latency,
   transport/compute delay, velocity limit and belt force limit. Measure pulley
   pitch circumference, gearing, cart mass, effective rail travel and metres per
   encoder count. Check the measured drive behavior against the simulation model.
4. Update `RigParams`, export again and run robustness, design screening and the
   closed-loop tests. A stable linear design alone does not validate the hardware.

## Integrate the drive and cart position

`readCartPositionMeters` must return a fresh, finite, centred cart position in
metres and reject stale CAN feedback or faults. Do not return zero as a placeholder.
`sendDriveVelocity` must enqueue an interrupt-safe, bounded-time moteus CAN-FD
velocity-mode command. Integrate the **delayed** acceleration as
`v_sp = clamp(v_sp + u * dt, +/-velocity_limit)`, then convert belt m/s to motor
revolutions/s using the pulley/gearing. Do not integrate again in the transport.
Disabling must send zero velocity and moteus stop/disable. Implement and verify a
drive-side command watchdog (at most 20 ms) and feedback/queue fault handling.
The two stubs intentionally contain no assumed moteus API or bus configuration.

The controller runs at 1 kHz in `IntervalTimer`: shift the position estimate when
the reference changes, compute `innovation = y - C*z` from the original estimate,
apply `z += L*innovation`, clamp `u = -K*z`, then predict `z = A*z + B*u` using
temporary storage. Its command-delay queue matches `Simulation`; the augmented
observer also models those in-flight commands. Keep the total measured sensor,
compute and CAN latency consistent with `compute_delay_steps`; do not silently
add another transport delay. Verify float32 behavior against Python before release.

## Calibration and bring-up

1. Start with drive power disabled and links secured in a jig that holds the
   chain in the exported configuration (`QP_CONFIGURATION`, e.g. links 1-3 up
   and link 4 down for `UUUD`). Home and
   centre the cart encoder. Verify all relative encoder signs and scale manually.
2. Press the calibration button or send `c` followed by newline to save all four
   current encoder offsets. Calibration disarms. The jig pose is taken as the
   equilibrium, so telemetry then reads `QP_THETA_EQ` (0 or pi per link). Confirm
   that and centred cart telemetry. The sketch cannot detect an incorrectly positioned jig.
3. Send `a` + newline to request arming. All four angle **deviations from the
   equilibrium** must remain within +/-3 degrees for 0.5 s; valid encoder/cart feedback and calibration are
   mandatory. The estimator and velocity setpoint reset at arming.
4. Test stop/disable (`d`), sensor faults, rail switches and the independent E-stop
   with the links restrained and conservative drive limits. The software disarms
   for any deviation from the equilibrium beyond 25 degrees, cart travel beyond the exported
   half-rail limit, invalid feedback, drive enqueue failure or timer overrun.
   Faults require another explicit `a` request; they do not auto-rearm.
5. Release from the jig only after signs, timing and stops are verified. Begin
   near the target equilibrium. The sketch cannot swing up or move between
   configurations. Send `r 0.2` + newline for a 0.2 m reference; requests clamp to
   80% of the rail half-length. Re-measure delay and velocity response on hardware.

CSV telemetry is sampled at 100 Hz and printed outside the ISR; samples may be
dropped if USB cannot keep up. Fields are tick, cart position/reference (m), four
absolute angles (rad, equilibrium plus deviation, so 0 or pi at rest), applied acceleration command (m/s²), integrated velocity
setpoint (m/s), ISR duration (microseconds), armed/calibrated flags and fault code.
Fault codes: 0 OK, 1 encoder, 2 cart feedback, 3 angle/rail, 4 drive transport,
5 ISR overrun, 6 nonfinite controller state. There is no acceleration or force
sensor implied by the acceleration-command telemetry. This is a bring-up skeleton,
not hardware-validated firmware; no Teensy toolchain or powered rig is exercised
by the Python checks.
