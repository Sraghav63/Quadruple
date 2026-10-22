import { LQRController, wrapAngle } from './controller';
import { Plant, SeededRandom } from './plant';
import { TransitionController, TransitionLibrary, type TransitionEntry } from './transition';
import type { Preset, RealityKnobs, SimStatus, SimulationApi } from './types';

export function DEFAULT_KNOBS(preset: Preset): RealityKnobs {
  return {
    frictionScale: 1, massScale: 1, sensorNoiseScale: 1,
    encoderBits: preset.rig.sensors.encoderBits, extraDelaySteps: 0,
    driveBandwidthScale: 1, accelLimit: preset.rig.drive.accelLimit,
  };
}

export function createSimulation(preset: Preset, seed = 0, transitions?: TransitionLibrary): SimulationApi {
  return new Simulation(preset, seed, transitions);
}

class Simulation implements SimulationApi {
  readonly rig;
  readonly knobs: RealityKnobs;
  private controllerOn = true;
  private readonly plant: Plant;
  private controller: LQRController;
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
  private readonly transitions: TransitionLibrary;
  private readonly trackers = new Map<string, TransitionController>();
  private mode: 'hold' | 'center' | 'transition' = 'hold';
  private target = '';
  private route: TransitionEntry[] = [];
  private tracker: TransitionController | null = null;
  private settled = 0;
  private requestToken = 0;
  private designProgress: number | null = null;
  private designMs = 0;
  private designMaxBatchMs = 0;
  private transitionError: string | null = null;

  private validateConfiguration(configuration: string): string {
    const cfg = configuration.trim().toUpperCase();
    if (!/^[UD]+$/.test(cfg) || cfg.length !== this.rig.links.length ||
        !Object.hasOwn(this.preset.configurations, cfg)) throw new RangeError('Invalid configuration');
    return cfg;
  }

  async requestConfiguration(configuration: string): Promise<boolean> {
    const goal = this.validateConfiguration(configuration);
    if (this.fallen || !this.controllerOn || this.mode !== 'hold') return false;
    const route = this.transitions.route(this.configuration, goal);
    this.transitionError = null;
    if (route === null) {
      this.transitionError = `No planned route from ${this.configuration} to ${goal}. Choose another target.`;
      return false;
    }
    const token = ++this.requestToken;
    this.xRef = 0; this.target = goal; this.route = route; this.settled = 0;
    this.mode = route.length ? 'center' : 'hold';
    if (!route.length) return true;
    this.designProgress = 0;
    this.designMs = this.designMaxBatchMs = 0;
    try {
      for (let i = 0; i < route.length; i++) {
        const e = route[i], key = `${e.start}-${e.goal}`;
        if (!this.trackers.has(key)) {
          const trajectory = await this.transitions.get(e);
          if (token !== this.requestToken) return false;
          const ctl = new TransitionController(this.rig, trajectory, this.preset.transitionConfig);
          await ctl.designAsync(f => { if (token === this.requestToken) this.designProgress = (i + f) / route.length; }, () => token !== this.requestToken);
          this.trackers.set(key, ctl);
          this.designMs += ctl.designMs;
          this.designMaxBatchMs = Math.max(this.designMaxBatchMs, ctl.maxBatchMs);
        }
      }
      if (token !== this.requestToken) return false;
      this.designProgress = null;
      return true;
    } catch (err) {
      if (token !== this.requestToken) return false;
      this.mode = 'hold'; this.target = this.configuration; this.route = []; this.designProgress = null;
      this.transitionError = `Could not prepare transition: ${err instanceof Error ? err.message : String(err)}. Try again.`;
      return false;
    }
  }

  private cancelTransition(): void {
    ++this.requestToken;
    this.mode = 'hold'; this.route = []; this.tracker = null; this.settled = 0;
    this.target = this.configuration; this.designProgress = null;
  }

  private inFlight(command: number): Float64Array {
    // At this point the queue still includes the command applied this period.
    // The new controller predicts the NEXT period: the newest d of pending[1:] + [command].
    // The true queue can be longer than the controller's model (extra latency knob).
    const d = this.rig.timing.computeDelaySteps, start = this.queueIndex + this.delay - (d - 1);
    return Float64Array.from({ length: d }, (_, i) => i < d - 1 ? this.pending[(start + i) % this.delay] : command);
  }

  private readyToStart(): boolean {
    const z = this.controller.physicalEstimate, n = this.rig.links.length;
    // Same thresholds as quad_pendulum.sim (CENTER_TOL ... SETTLE_TIME).
    let still = Math.abs(z[0]) < 0.05 && Math.abs(z[1]) < 0.6;
    for (let i = 0; i < n; i++) still &&= Math.abs(z[3 + i]) < 6 * Math.PI / 180 && Math.abs(z[3 + n + i]) < 1.5;
    this.settled = still ? this.settled + 1 : 0;
    return this.settled >= Math.max(1, Math.round(0.0 * this.rig.timing.controlRate));
  }

  get configuration(): string { return this.controller.matrices.configuration; }

  setConfiguration(configuration: string, opts?: { randomTiltDeg?: number; theta?: number[] }): void {
    const cfg = this.validateConfiguration(configuration);
    this.controller = new LQRController(this.preset.configurations[cfg].controller);
    this.resetAtConfiguration(opts);
  }

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
    this.cancelTransition();
    this.plant.setDriveEnabled(false);
    this.pending.fill(0);
    this.queueIndex = 0;
    this.command = 0;
  }

  constructor(readonly preset: Preset, seed: number, transitions?: TransitionLibrary) {
    this.rig = preset.rig;
    this.knobs = DEFAULT_KNOBS(preset);
    this.plant = new Plant(this.rig, this.knobs, seed);
    this.controller = new LQRController(preset.configurations['D'.repeat(this.rig.links.length)].controller);
    this.transitions = transitions ?? new TransitionLibrary(preset.transitions ?? [], async entry => {
      const base = import.meta.env.BASE_URL;
      const url = `${base.endsWith('/') ? base : `${base}/`}presets/${encodeURIComponent(preset.id)}/transitions/${encodeURIComponent(entry.file)}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status} loading ${entry.file}`);
      return response.json();
    });
    this.rng = new SeededRandom(seed);
    this.delay = Math.max(0, Math.round(this.rig.timing.computeDelaySteps));
    this.pending = new Float64Array(this.delay + 10);
    this.initialTheta = new Float64Array(this.rig.links.length);
    this.points = Array.from({ length: this.rig.links.length + 1 }, () => [0, 0]);
    this.snapshot = {
      mode: 'hold', target: this.configuration, route: [], currentEdge: null, transitionProgress: null,
      designProgress: null, designMs: 0, designMaxBatchMs: 0, transitionError: null, xEstimate: 0,
      configuration: this.configuration, thetaDeviation: new Float64Array(this.rig.links.length),
      t: 0, x: 0, v: 0, a: 0, theta: new Float64Array(this.rig.links.length),
      omega: new Float64Array(this.rig.links.length), thetaEstimate: new Float64Array(this.rig.links.length),
      xRef: 0, command: 0, force: 0, driveEnabled: true, vSetpoint: 0,
      fallen: false, hitRailEnd: false, controllerEnabled: true,
    };
    this.reset();
  }

  reset(opts?: { randomTiltDeg?: number; theta?: number[] }): void {
    this.controller = new LQRController(this.preset.configurations['D'.repeat(this.rig.links.length)].controller);
    this.xRef = 0;
    this.resetAtConfiguration(opts);
  }

  private resetAtConfiguration(opts?: { randomTiltDeg?: number; theta?: number[] }): void {
    this.cancelTransition();
    this.transitionError = null;
    this.designMs = this.designMaxBatchMs = 0;
    const tilt = opts?.randomTiltDeg ?? 0;
    if (!Number.isFinite(tilt) || tilt < 0) throw new RangeError('Invalid random tilt');
    for (let i = 0; i < this.initialTheta.length; i++) {
      this.initialTheta[i] = this.controller.matrices.thetaEq[i] + (2 * this.rng.uniform() - 1) * tilt * Math.PI / 180;
    }
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
    if (this.mode !== 'hold') return;
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
    if (!armed) this.controller.reset(measurement, this.xRef);
    else if (this.mode === 'transition' && this.tracker) {
      command = this.tracker.step(measurement);
      if (this.tracker.done) {
        const edge = this.route.shift()!;
        this.controller = new LQRController(this.preset.configurations[edge.goal].controller);
        this.controller.seed(this.tracker.handoff(), this.inFlight(command));
        this.tracker = null; this.mode = this.route.length ? 'center' : 'hold'; this.settled = 0;
      }
    } else {
      command = this.controller.step(measurement, this.mode === 'center' ? 0 : this.xRef);
      if (this.mode === 'center' && this.readyToStart() && this.designProgress === null) {
        const e = this.route[0];
        this.tracker = this.trackers.get(`${e.start}-${e.goal}`)!;
        const z = this.controller.physicalEstimate;
        this.tracker.begin(z, this.inFlight(command), z[0]);
        this.mode = 'transition';
      }
    }
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
    for (let i = 0; i < this.plant.n; i++) {
      const target = this.tracker ? this.tracker.nominal[3 + i][Math.min(this.tracker.k, this.tracker.steps)] : this.controller.matrices.thetaEq[i];
      if (Math.abs(wrapAngle(this.plant.theta[i] - target)) > 0.6) fallen = true;
    }
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
    const m = this.controller.matrices;
    s.configuration = this.configuration;
    s.mode = this.mode; s.target = this.target;
    s.route = this.route.length ? [this.route[0].start, ...this.route.map(e => e.goal)] : [this.configuration];
    s.currentEdge = this.route.length ? `${this.route[0].start} → ${this.route[0].goal}` : null;
    s.transitionProgress = this.tracker?.progress ?? null; s.designProgress = this.designProgress;
    s.designMs = this.designMs; s.designMaxBatchMs = this.designMaxBatchMs; s.transitionError = this.transitionError;
    const tracker = this.tracker, k = tracker ? Math.min(tracker.k, tracker.steps) : 0;
    s.xEstimate = tracker ? tracker.nominal[0][k] + tracker.xOffset + tracker.z[0] : this.controller.z[0] + this.xRef;
    for (let i = 0; i < p.n; i++) {
      const target = tracker ? tracker.nominal[3 + i][k] : m.thetaEq[i];
      s.thetaEstimate[i] = target + (tracker ? tracker.z[3 + i] : this.controller.z[m.thetaIndex + i]);
      s.thetaDeviation[i] = wrapAngle(p.theta[i] - target);
    }
    s.xRef = tracker ? tracker.nominal[0][k] + tracker.xOffset : this.xRef; s.command = this.command; s.force = p.force;
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
