import type { TransitionEntry, TransitionConfig } from "./transition";
// Shared contract between the simulation core (src/sim) and the UI (src/ui).
// Preset JSON is produced by scripts/export_web.py from the Python package, so
// field names mirror quad_pendulum.params (snake_case -> camelCase).
//
// Conventions (same as the Python code and the firmware):
// - jointAngles q_i: link i relative to link i-1 (link 0 = cart), 0 = straight up,
//   positive tilts toward +x. Absolute angles theta_i = q_1 + ... + q_i.
// - The controller outputs a cart acceleration command (m/s^2). The servo drive
//   integrates it into a velocity setpoint and closes a PI velocity loop -> force.

export interface LinkParams {
  length: number; // m, hinge to hinge
  rodMass: number; // kg, uniform rod
  topMass: number; // kg, point mass at the upper end (next joint's hardware)
}

export interface RigParams {
  links: LinkParams[];
  joints: { coulomb: number; viscous: number }; // N m, N m s/rad (relative joint motion)
  cart: { mass: number; railHalfLength: number; friction: number; viscous: number };
  drive: {
    accelLimit: number; // m/s^2
    velocityLimit: number; // m/s
    forceLimit: number; // N
    velocityBandwidth: number; // rad/s
    modelLag: number; // s (what the controller designed against)
  };
  sensors: { encoderBits: number; encoderNoise: number; cartResolution: number; cartNoise: number };
  timing: { controlRate: number; physicsDt: number; computeDelaySteps: number };
  gravity: number;
}

/** Fixed-matrix runtime, identical to firmware/teensy and LQRController.step. */
export interface ControllerMatrices {
  configuration: string; // U/D per link, bottom first
  thetaEq: number[]; // absolute target angles, rad
  nz: number;
  nx: number;
  delay: number;
  ny: number; // 1 + nLinks
  dt: number;
  accelLimit: number;
  thetaIndex: number; // z[thetaIndex .. thetaIndex+n-1] are deviations from thetaEq
  stateLayout: string;
  K: number[]; // (nz)
  L: number[][]; // (nz x ny)
  A: number[][]; // (nz x nz)
  B: number[]; // (nz)
  C: number[][]; // (ny x nz)
  closedLoopRadius: number;
  observerRadius: number;
}

export interface Preset {
  transitions?: TransitionEntry[];
  transitionConfig?: TransitionConfig;
  id: string;
  name: string;
  description: string;
  unstablePoles: number[]; // rad/s, fastest first
  rig: RigParams;
  controller: ControllerMatrices;
  configurations: Record<string, { controller: ControllerMatrices; unstablePoles: number[] }>;
}

export interface PresetSummary {
  id: string;
  name: string;
  description: string;
}

/** What the microcontroller reads each period (quantised). */
export interface Measurement {
  t: number;
  cartPosition: number;
  jointAngles: Float64Array;
}

/**
 * "Real world" knobs the UI can change live WITHOUT redesigning the controller.
 * The controller keeps the preset's model; these change only the true plant and
 * sensors, which is exactly the model-mismatch question a real build faces.
 * All default to 1 (or the preset value).
 */
export interface RealityKnobs {
  frictionScale: number; // multiplies joint coulomb + viscous
  massScale: number; // multiplies link rod + top masses
  sensorNoiseScale: number; // multiplies encoder noise (0 = ideal)
  encoderBits: number; // overrides rig.sensors.encoderBits
  extraDelaySteps: number; // additional control periods of latency (0..10)
  driveBandwidthScale: number; // multiplies drive.velocityBandwidth
  accelLimit: number; // overrides the clamp applied by the drive (m/s^2)
}

export interface SimStatus {
  mode: 'hold' | 'center' | 'transition';
  target: string;
  route: string[];
  currentEdge: string | null;
  transitionProgress: number | null;
  designProgress: number | null;
  designMs: number;
  designMaxBatchMs: number;
  transitionError: string | null;
  xEstimate: number;

  configuration: string;
  t: number; // s
  x: number; // cart position, m
  v: number; // m/s
  a: number; // actual cart acceleration, m/s^2
  theta: Float64Array; // absolute link angles, rad
  thetaDeviation: Float64Array; // wrapped angle errors from the target, rad
  omega: Float64Array; // rad/s
  thetaEstimate: Float64Array; // controller's estimate, rad
  xRef: number;
  command: number; // accel command applied this period, m/s^2
  force: number; // drive force, N
  driveEnabled: boolean;
  vSetpoint: number; // drive velocity setpoint, m/s; zero while disabled
  fallen: boolean;
  hitRailEnd: boolean;
  controllerEnabled: boolean;
}

/**
 * Simulation API used by the UI. One `step()` = one control period (1 ms).
 * The UI calls `advance(wallSeconds * speed)` once per animation frame.
 */
export interface SimulationApi {
  readonly preset: Preset;
  readonly rig: RigParams; // nominal (preset) rig
  readonly knobs: RealityKnobs;
  readonly configuration: string;
  setConfiguration(configuration: string, opts?: { randomTiltDeg?: number; theta?: number[] }): void; // explicit equilibrium jump, for parity fixtures
  requestConfiguration(configuration: string): Promise<boolean>; // live, without resetting
  controllerEnabled: boolean;
  reset(opts?: { randomTiltDeg?: number; theta?: number[] }): void;
  setReference(x: number): void; // clipped to +-0.8 rail
  push(link: number, forceN: number, durationS?: number): void; // horizontal at the link top
  setKnobs(k: Partial<RealityKnobs>): void;
  step(): void;
  advance(simSeconds: number, maxSteps?: number): number; // returns steps taken
  status(): SimStatus;
  /** (x, z) of each hinge and the tip, world metres; length n+1. */
  linkPoints(): Array<[number, number]>;
}
