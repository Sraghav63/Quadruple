import type { Measurement, RigParams } from './types';
import { wrapAngle } from './controller';
import { ChainDynamics } from './linear';
import { add, dareDoubling, expm, eye, mul, scaled, solve, sym, transpose, zeros, type Matrix } from './matrix';

export interface TransitionEntry { start: string; goal: string; duration: number; file: string }
export interface TrajectoryData {
  start: string; goal: string; duration: number; knots: number;
  x: number[]; v: number[]; a: number[]; jerk: number[];
  theta: number[][]; omega: number[][]; alpha: number[][];
}
// Python/NumPy round-to-even, including duration/dt exactly at a half step.
function roundEven(x: number): number { const f = Math.floor(x); return x - f === 0.5 ? f + f % 2 : Math.round(x); }
export class Trajectory {
  constructor(readonly data: TrajectoryData) {}
  sample(dt: number, lag: number): { nominal: Matrix; commands: Float64Array; steps: number } {
    const d = this.data, steps = roundEven(d.duration / dt), h = d.duration / (d.x.length - 1);
    const pairs = [[d.x, d.v], [d.v, d.a], [d.a, d.jerk], ...d.theta.map((row, j) => [row, d.omega[j]]), ...d.omega.map((row, j) => [row, d.alpha[j]])];
    const nominal = zeros(pairs.length, steps + 1), commands = new Float64Array(steps + 1);
    for (let k = 0; k <= steps; k++) {
      const s = Math.max(0, Math.min(k * dt / h, d.x.length - 1 - 1e-12)), i = Math.floor(s), f = s - i, f2 = f * f, f3 = f2 * f;
      pairs.forEach(([p, m], j) => {
        nominal[j][k] = (2 * f3 - 3 * f2 + 1) * p[i] + (f3 - 2 * f2 + f) * m[i] * h + (-2 * f3 + 3 * f2) * p[i + 1] + (f3 - f2) * m[i + 1] * h;
      });
      commands[k] = nominal[2][k] + lag * (d.jerk[i] * (1 - f) + d.jerk[i + 1] * f);
    }
    return { nominal, commands, steps };
  }
}

/** Duration-weighted sparse directed graph. Files load only on a requested route. */
export class TransitionLibrary {
  private loaded = new Map<string, Promise<Trajectory>>();
  constructor(readonly entries: TransitionEntry[], private loader: (entry: TransitionEntry) => Promise<TrajectoryData>) {}
  route(start: string, goal: string): TransitionEntry[] | null {
    if (start === goal) return [];
    const best = new Map([[start, 0]]), prev = new Map<string, TransitionEntry>();
    const frontier: Array<[number, string]> = [[0, start]];
    while (frontier.length) {
      frontier.sort((a, b) => a[0] - b[0] || a[1].localeCompare(b[1]));
      const [cost, node] = frontier.shift()!;
      if (node === goal) break;
      if (cost > best.get(node)!) continue;
      for (const e of this.entries) if (e.start === node && cost + e.duration < (best.get(e.goal) ?? Infinity)) {
        best.set(e.goal, cost + e.duration); prev.set(e.goal, e); frontier.push([cost + e.duration, e.goal]);
      }
    }
    if (!best.has(goal)) return null;
    const path: TransitionEntry[] = [];
    for (let node = goal; node !== start;) { const e = prev.get(node)!; path.push(e); node = e.start; }
    return path.reverse();
  }
  get(entry: TransitionEntry): Promise<Trajectory> {
    const key = `${entry.start}-${entry.goal}`;
    let p = this.loaded.get(key);
    if (!p) {
      p = this.loader(entry).then(data => new Trajectory(data)); this.loaded.set(key, p);
      void p.catch(() => this.loaded.delete(key));
    }
    return p;
  }
}

export interface TransitionConfig {
  cartPositionTol: number; cartVelocityTol: number; angleTol: number; rateTol: number;
  jointAngleTol: number | null; accelTol: number; cartAccelNoise: number; linkAccelNoise: number; fdStep: number;
}
export const DEFAULT_TRANSITION_CONFIG: TransitionConfig = {
  cartPositionTol: 0.08, cartVelocityTol: 1, angleTol: 0.05, rateTol: 0.5,
  jointAngleTol: 0.01, accelTol: 1, cartAccelNoise: 0.06, linkAccelNoise: 2.5, fdStep: 1e-6,
};

export class TransitionController {
  readonly n: number;
  readonly ns: number;
  readonly nz: number;
  readonly delay: number;
  readonly dt: number;
  readonly nominal: Matrix;
  readonly uNominal: Float64Array;
  readonly uIssue: Float64Array;
  readonly steps: number;
  readonly az: Matrix[] = [];
  readonly bz: Matrix[] = [];
  readonly gains: Float64Array[] = [];
  readonly c: Matrix;
  readonly measCov: Matrix;
  readonly processCov: Matrix;
  z: Float64Array;
  p: Matrix;
  k = 0;
  xOffset = 0;
  designMs = 0;
  maxBatchMs = 0;

  constructor(readonly rig: RigParams, readonly trajectory: Trajectory, readonly config: TransitionConfig = DEFAULT_TRANSITION_CONFIG) {
    this.n = rig.links.length; this.ns = 3 + 2 * this.n; this.delay = rig.timing.computeDelaySteps;
    this.nz = this.ns + this.delay; this.dt = 1 / rig.timing.controlRate;
    const sampled = trajectory.sample(this.dt, rig.drive.modelLag);
    this.nominal = sampled.nominal; this.uNominal = sampled.commands; this.steps = sampled.steps;
    this.uIssue = Float64Array.from({ length: this.steps }, (_, k) => this.uNominal[k + this.delay] ?? 0);
    this.c = zeros(1 + this.n, this.nz); this.c[0][0] = 1;
    for (let i = 0; i < this.n; i++) this.c[i + 1][3 + i] = 1;
    const sensors = rig.sensors, enc = (2 * Math.PI / 2 ** sensors.encoderBits) ** 2 / 12 + sensors.encoderNoise ** 2;
    this.measCov = zeros(1 + this.n); this.measCov[0][0] = sensors.cartResolution ** 2 / 12 + sensors.cartNoise ** 2;
    for (let i = 1; i <= this.n; i++) for (let j = 1; j <= this.n; j++) this.measCov[i][j] = enc * Math.min(i, j);
    this.processCov = eye(this.nz, 1e-12); this.processCov[2][2] += config.cartAccelNoise ** 2;
    for (let i = 3 + this.n; i < this.ns; i++) this.processCov[i][i] += (config.linkAccelNoise * this.dt) ** 2;
    this.z = new Float64Array(this.nz); this.p = eye(this.nz, 1e-6);
  }

  /** Each yield bounds a unit of work; the browser driver budgets batches to 4 ms. */
  *design(): Generator<void> {
    const { n, ns, nz, delay: d, rig, config: cfg } = this, dynamics = new ChainDynamics(rig);
    for (let k = 0; k < this.steps; k++) {
      const theta = this.nominal.slice(3, 3 + n).map(row => row[k]), omega = this.nominal.slice(3 + n).map(row => row[k]);
      const jac = dynamics.jacobian(theta, omega, this.nominal[2][k], cfg.fdStep);
      const m = zeros(ns + 1); m[0][1] = m[1][2] = 1; m[2][2] = -1 / rig.drive.modelLag; m[2][ns] = 1 / rig.drive.modelLag;
      for (let i = 0; i < n; i++) {
        m[3 + i][3 + n + i] = 1; m[3 + n + i][2] = jac[i][0];
        for (let j = 3; j < ns; j++) m[3 + n + i][j] = jac[i][j - 2];
      }
      const e = expm(scaled(m, this.dt)), a = zeros(nz), b = zeros(nz, 1);
      for (let i = 0; i < ns; i++) { a[i].set(e[i].subarray(0, ns)); if (d) a[i][ns] = e[i][ns]; else b[i][0] = e[i][ns]; }
      if (d) { for (let i = 0; i < d - 1; i++) a[ns + i][ns + i + 1] = 1; b[nz - 1][0] = 1; }
      this.az[k] = a; this.bz[k] = b;
      yield;
    }
    const q = zeros(nz); q[0][0] = 1 / cfg.cartPositionTol ** 2; q[1][1] = 1 / cfg.cartVelocityTol ** 2;
    for (let i = 0; i < n; i++) {
      q[3 + i][3 + i] = 1 / cfg.angleTol ** 2; q[3 + n + i][3 + n + i] = 1 / cfg.rateTol ** 2;
    }
    if (cfg.jointAngleTol) for (let i = 0; i < n; i++) {
      q[3 + i][3 + i] += 1 / cfg.jointAngleTol ** 2;
      if (i) { q[2 + i][2 + i] += 1 / cfg.jointAngleTol ** 2; q[2 + i][3 + i] -= 1 / cfg.jointAngleTol ** 2; q[3 + i][2 + i] -= 1 / cfg.jointAngleTol ** 2; }
    }
    const r = 1 / cfg.accelTol ** 2;
    let p = dareDoubling(this.az[this.steps - 1], this.bz[this.steps - 1], q, [new Float64Array([r])]);
    yield;
    for (let k = this.steps - 1; k >= 0; k--) {
      const a = this.az[k], b = this.bz[k], bt = transpose(b), btp = mul(bt, p);
      const gain = scaled(mul(btp, a), 1 / (r + mul(btp, b)[0][0]));
      this.gains[k] = gain[0];
      p = sym(add(q, mul(mul(transpose(a), p), add(a, mul(b, gain), -1))));
      yield;
    }
  }
  async designAsync(onProgress?: (fraction: number) => void, cancelled = () => false): Promise<void> {
    const start = performance.now(), generator = this.design();
    let count = 0, done = false;
    while (!done) {
      if (cancelled()) throw new Error('Transition request cancelled');
      const batch = performance.now();
      do { done = !!generator.next().done; count++; } while (!done && performance.now() - batch < 4);
      this.maxBatchMs = Math.max(this.maxBatchMs, performance.now() - batch);
      onProgress?.(Math.min(1, count / (2 * this.steps + 1)));
      if (!done) await new Promise<void>(resolve => setTimeout(resolve, 0));
    }
    this.designMs = performance.now() - start;
  }
  begin(deviation: ArrayLike<number>, inFlight: ArrayLike<number>, xOffset = 0): void {
    this.k = 0; this.xOffset = xOffset; this.z.fill(0);
    for (let i = 0; i < this.ns; i++) this.z[i] = deviation[i];
    this.z[0] -= xOffset;
    for (let i = 0; i < this.delay; i++) this.z[this.ns + i] = inFlight[i] - this.uNominal[i];
    this.p = eye(this.nz, 1e-6);
  }
  get done(): boolean { return this.k >= this.steps; }
  get progress(): number { return Math.min(1, this.k / this.steps); }
  step(meas: Measurement): number {
    const k = Math.min(this.k, this.steps - 1), y = zeros(1 + this.n, 1);
    y[0][0] = meas.cartPosition - this.nominal[0][k] - this.xOffset;
    let theta = 0;
    for (let i = 0; i < this.n; i++) { theta += meas.jointAngles[i]; y[1 + i][0] = wrapAngle(theta - this.nominal[3 + i][k]); }
    const ct = transpose(this.c), pct = mul(this.p, ct);
    // Python multiplies by inv(S); solving the transposed system avoids an explicit inverse.
    const gain = transpose(solve(transpose(add(mul(this.c, pct), this.measCov)), transpose(pct)));
    const z = this.z.slice(); // correction before the command and prediction
    const innovation = add(y, mul(this.c, Array.from(z, v => new Float64Array([v]))), -1);
    const correction = mul(gain, innovation);
    for (let i = 0; i < this.nz; i++) z[i] += correction[i][0];
    const p = mul(add(eye(this.nz), mul(gain, this.c), -1), this.p);
    let u = this.uIssue[k]; for (let i = 0; i < this.nz; i++) u -= this.gains[k][i] * z[i];
    u = Math.max(-this.rig.drive.accelLimit, Math.min(this.rig.drive.accelLimit, u));
    const next = mul(this.az[k], Array.from(z, v => new Float64Array([v])));
    for (let i = 0; i < this.nz; i++) this.z[i] = next[i][0] + this.bz[k][i][0] * (u - this.uIssue[k]);
    this.p = add(mul(mul(this.az[k], p), transpose(this.az[k])), this.processCov);
    this.k++;
    return u;
  }
  get thetaEstimate(): Float64Array {
    return Float64Array.from({ length: this.n }, (_, i) => this.nominal[3 + i][Math.min(this.k, this.steps)] + this.z[3 + i]);
  }
  handoff(): Float64Array { const dev = this.z.slice(0, this.ns); dev[0] += this.xOffset; return dev; }
}
