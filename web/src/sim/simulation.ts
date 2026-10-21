import { LQRController } from './controller';
import { Plant, SeededRandom } from './plant';
import type { Preset, RealityKnobs, SimStatus, SimulationApi } from './types';

export function DEFAULT_KNOBS(preset: Preset): RealityKnobs {
  return {
    frictionScale: 1, massScale: 1, sensorNoiseScale: 1,
    encoderBits: preset.rig.sensors.encoderBits, extraDelaySteps: 0,
    driveBandwidthScale: 1, accelLimit: preset.rig.drive.accelLimit,
  };
}

export function createSimulation(preset: Preset, seed = 0): SimulationApi {
  return new Simulation(preset, seed);
}

class Simulation implements SimulationApi {
  readonly rig;
  readonly knobs: RealityKnobs;
  private controllerOn = true;
  private readonly plant: Plant;
  private readonly controller: LQRController;
  private readonly rng: SeededRandom;
  private readonly pending: Float64Array;
  private delay: number;
  private queueIndex = 0;
  private remainder = 0;
  private xRef = 0;
  private command = 0;
  private fallen = false;
  private readonly snapshot: SimStatus;
  private readonly points: Array<[number, number]>;
  private readonly initialTheta: Float64Array;

  get controllerEnabled(): boolean { return this.controllerOn; }
  set controllerEnabled(enabled: boolean) {
    if (enabled === this.controllerOn) return;
    this.controllerOn = enabled;
    if (enabled && !this.fallen) {
      this.controller.reset(this.plant.measure(), this.xRef);
      this.plant.setDriveEnabled(true);
    } else this.disableDrive();
  }

  private disableDrive(): void {
    this.plant.setDriveEnabled(false);
    this.pending.fill(0);
    this.queueIndex = 0;
    this.command = 0;
  }

  constructor(readonly preset: Preset, seed: number) {
    this.rig = preset.rig;
    this.knobs = DEFAULT_KNOBS(preset);
    this.plant = new Plant(this.rig, this.knobs, seed);
    this.controller = new LQRController(preset.controller);
    this.rng = new SeededRandom(seed);
    this.delay = Math.max(0, Math.round(this.rig.timing.computeDelaySteps));
    this.pending = new Float64Array(this.delay + 10);
    this.initialTheta = new Float64Array(this.rig.links.length);
    this.points = Array.from({ length: this.rig.links.length + 1 }, () => [0, 0]);
    this.snapshot = {
      t: 0, x: 0, v: 0, a: 0, theta: new Float64Array(this.rig.links.length),
      omega: new Float64Array(this.rig.links.length), thetaEstimate: new Float64Array(this.rig.links.length),
      xRef: 0, command: 0, force: 0, driveEnabled: true, vSetpoint: 0,
      fallen: false, hitRailEnd: false, controllerEnabled: true,
    };
    this.reset();
  }

  reset(opts?: { randomTiltDeg?: number; theta?: number[] }): void {
    const tilt = opts?.randomTiltDeg ?? 0;
    if (!Number.isFinite(tilt) || tilt < 0) throw new RangeError('Invalid random tilt');
    for (let i = 0; i < this.initialTheta.length; i++) this.initialTheta[i] = (2 * this.rng.uniform() - 1) * tilt * Math.PI / 180;
    this.plant.reset(opts?.theta ?? this.initialTheta);
    // Keep the requested reference across resets, initializing in that frame.
    this.controller.reset(this.plant.measure(), this.xRef);
    this.pending.fill(0);
    this.queueIndex = 0;
    this.remainder = this.command = 0;
    this.fallen = false;
    this.controllerOn = true;
  }

  setReference(x: number): void {
    if (!Number.isFinite(x)) throw new RangeError('Reference must be finite');
    const limit = 0.8 * this.rig.cart.railHalfLength;
    this.xRef = Math.max(-limit, Math.min(limit, x));
  }

  push(link: number, forceN: number, durationS = 0.02): void { this.plant.push(link, forceN, durationS); }

  setKnobs(k: Partial<RealityKnobs>): void {
    const next = { ...this.knobs, ...k };
    if (Object.values(next).some(v => !Number.isFinite(v)) || next.massScale <= 0 ||
        next.frictionScale < 0 || next.sensorNoiseScale < 0 || next.driveBandwidthScale <= 0 || next.accelLimit < 0 ||
        next.encoderBits < 1 || next.encoderBits > 32 || next.extraDelaySteps < 0 || next.extraDelaySteps > 10) {
      throw new RangeError('Invalid reality knobs');
    }
    next.encoderBits = Math.round(next.encoderBits);
    next.extraDelaySteps = Math.round(next.extraDelaySteps);
    Object.assign(this.knobs, next);
    const delay = this.rig.timing.computeDelaySteps + next.extraDelaySteps;
    if (delay !== this.delay) {
      // A latency change clears commands already in flight, then uses the new queue.
      this.delay = delay;
      this.pending.fill(0);
      this.queueIndex = 0;
    }
    this.plant.configure();
  }

  step(): void {
    const measurement = this.plant.measure();
    const armed = this.controllerEnabled && !this.fallen;
    let command = 0;
    if (armed) command = this.controller.step(measurement, this.xRef);
    else this.controller.reset(measurement, this.xRef);
    let applied = command;
    if (this.delay > 0) {
      applied = this.pending[this.queueIndex];
      this.pending[this.queueIndex] = command;
      this.queueIndex = (this.queueIndex + 1) % this.delay;
    }
    // Safety disarm also suppresses commands still in flight.
    this.command = armed ? applied : 0;
    this.plant.step(this.command);
    let fallen = this.plant.hitRailEnd;
    for (let i = 0; i < this.plant.n; i++) if (Math.abs(this.plant.theta[i]) > 0.6) fallen = true;
    if (fallen && !this.fallen) {
      this.fallen = true;
      this.disableDrive();
    }
  }

  advance(simSeconds: number, maxSteps = 100): number {
    if (!Number.isFinite(simSeconds) || simSeconds < 0 || !Number.isFinite(maxSteps) || maxSteps < 0) throw new RangeError('Invalid advance');
    const rate = this.rig.timing.controlRate;
    this.remainder += simSeconds * rate;
    const steps = Math.min(Math.floor(this.remainder + 1e-10), Math.floor(maxSteps));
    for (let i = 0; i < steps; i++) this.step();
    this.remainder = Math.max(0, this.remainder - steps);
    return steps;
  }

  status(): SimStatus {
    const s = this.snapshot, p = this.plant;
    s.t = p.t; s.x = p.q[0]; s.v = p.velocity[0]; s.a = p.a;
    s.theta.set(p.theta); s.omega.set(p.omega);
    for (let i = 0; i < p.n; i++) s.thetaEstimate[i] = this.controller.z[this.preset.controller.thetaIndex + i];
    s.xRef = this.xRef; s.command = this.command; s.force = p.force;
    s.driveEnabled = p.driveEnabled; s.vSetpoint = p.vSetpoint;
    s.fallen = this.fallen; s.hitRailEnd = p.hitRailEnd; s.controllerEnabled = this.controllerEnabled;
    return s;
  }

  linkPoints(): Array<[number, number]> {
    let x = this.plant.q[0], z = 0.07;
    this.points[0][0] = x; this.points[0][1] = z;
    for (let i = 0; i < this.plant.n; i++) {
      x += this.rig.links[i].length * Math.sin(this.plant.theta[i]);
      z += this.rig.links[i].length * Math.cos(this.plant.theta[i]);
      this.points[i + 1][0] = x; this.points[i + 1][1] = z;
    }
    return this.points;
  }
}
