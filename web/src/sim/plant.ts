import type { Measurement, RealityKnobs, RigParams } from './types';

// MuJoCo default solref=(0.02, 1), solimp=(0.9, 0.95, ...).
const FRICTION_VELOCITY_GAIN = 2 / (0.95 * 0.02);
const clamp = (v: number, limit: number): number => Math.max(-limit, Math.min(limit, v));

/** Mulberry32 and Box-Muller; independent of platform random state. */
export class SeededRandom {
  private state: number;
  private spare = 0;
  private hasSpare = false;
  constructor(seed = 0) { this.state = seed >>> 0; }
  uniform(): number {
    let t = this.state = (this.state + 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  normal(): number {
    if (this.hasSpare) { this.hasSpare = false; return this.spare; }
    const r = Math.sqrt(-2 * Math.log(1 - this.uniform()));
    const a = 2 * Math.PI * this.uniform();
    this.spare = r * Math.sin(a);
    this.hasSpare = true;
    return r * Math.cos(a);
  }
}

/** In-place Cholesky factorization and solve; matrices are row-major. */
function factor(a: Float64Array, l: Float64Array, n: number): void {
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let v = a[i * n + j];
    for (let k = 0; k < j; k++) v -= l[i * n + k] * l[j * n + k];
    l[i * n + j] = i === j ? Math.sqrt(v) : v / l[j * n + j];
  }
}
function solve(l: Float64Array, b: Float64Array, out: Float64Array, n: number): void {
  for (let i = 0; i < n; i++) {
    let v = b[i];
    for (let j = 0; j < i; j++) v -= l[i * n + j] * out[j];
    out[i] = v / l[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let v = out[i];
    for (let j = i + 1; j < n; j++) v -= l[j * n + i] * out[j];
    out[i] = v / l[i * n + i];
  }
}

/**
 * Exact planar rigid-body mechanics in relative hinge coordinates. Integration
 * mirrors MuJoCo implicitfast: implicit viscous damping, semi-implicit position.
 * Dry friction is a bounded, coupled acceleration constraint, not a sign/tanh
 * force. Default MuJoCo solref/solimp give aref=-2*v/(.95*.02), R=invweight0/9.
 */
export class Plant {
  readonly n: number;
  readonly q: Float64Array;
  readonly velocity: Float64Array;
  readonly theta: Float64Array;
  readonly omega: Float64Array;
  t = 0;
  force = 0;
  a = 0;
  hitRailEnd = false;
  private driveOn = true;
  private setpoint = 0;
  private integral = 0;
  private readonly mp: Float64Array;
  private readonly moment: Float64Array;
  private readonly inertia: Float64Array;
  private totalMass = 0;
  private kp = 0;
  private ki = 0;
  private readonly mass: Float64Array;
  private readonly chol: Float64Array;
  private readonly inverse: Float64Array;
  private readonly rhs: Float64Array;
  private readonly acceleration: Float64Array;
  private readonly scratch: Float64Array;
  private readonly unit: Float64Array;
  private readonly sin: Float64Array;
  private readonly cos: Float64Array;
  private readonly regularization: Float64Array;
  private readonly friction: Float64Array;
  private readonly frictionForce: Float64Array;
  private readonly damping: Float64Array;
  private readonly measurement: Measurement;
  private readonly rng: SeededRandom;
  private readonly pushes: Array<{ link: number; force: number; end: number }> = [];
  private readonly substeps: number;

  get driveEnabled(): boolean { return this.driveOn; }
  get vSetpoint(): number { return this.setpoint; }

  /** Enabling a real velocity drive starts at measured speed, with no stored torque. */
  setDriveEnabled(enabled: boolean): void {
    if (enabled === this.driveOn) return;
    this.driveOn = enabled;
    this.setpoint = enabled ? this.velocity[0] : 0;
    this.integral = this.force = 0;
    this.evaluateAcceleration(0, 0);
  }

  constructor(readonly rig: RigParams, readonly knobs: RealityKnobs, seed = 0) {
    this.n = rig.links.length;
    const n = this.n, d = n + 1;
    this.q = new Float64Array(d);
    this.velocity = new Float64Array(d);
    this.theta = new Float64Array(n);
    this.omega = new Float64Array(n);
    this.mp = new Float64Array(n * n);
    this.moment = new Float64Array(n);
    this.inertia = new Float64Array(n);
    this.mass = new Float64Array(d * d);
    this.chol = new Float64Array(d * d);
    this.inverse = new Float64Array(d * d);
    this.rhs = new Float64Array(d);
    this.acceleration = new Float64Array(d);
    this.scratch = new Float64Array(d);
    this.unit = new Float64Array(d);
    this.sin = new Float64Array(n);
    this.cos = new Float64Array(n);
    this.regularization = new Float64Array(d);
    this.friction = new Float64Array(d);
    this.frictionForce = new Float64Array(d);
    this.damping = new Float64Array(d);
    this.measurement = { t: 0, cartPosition: 0, jointAngles: new Float64Array(n) };
    this.rng = new SeededRandom(seed);
    const substeps = 1 / rig.timing.controlRate / rig.timing.physicsDt;
    if (!Number.isFinite(substeps) || substeps < 1 || Math.abs(substeps - Math.round(substeps)) > 1e-9) {
      throw new Error('physicsDt must divide the control period');
    }
    this.substeps = Math.round(substeps);
    this.configure();
    this.reset();
  }

  /** Recompute true masses and servo gains only; never changes controller matrices. */
  configure(): void {
    const { rig, knobs, n, mp, moment, inertia } = this;
    mp.fill(0); moment.fill(0);
    this.totalMass = rig.cart.mass;
    for (let k = 0; k < n; k++) {
      const link = rig.links[k];
      const rod = link.rodMass * knobs.massScale, top = link.topMass * knobs.massScale;
      this.totalMass += rod + top;
      inertia[k] = rod * link.length * link.length / 12;
      for (let i = 0; i <= k; i++) {
        const ci = rig.links[i].length * (i === k ? 0.5 : 1);
        moment[i] += rod * ci + top * rig.links[i].length;
        for (let j = 0; j <= k; j++) {
          const cj = rig.links[j].length * (j === k ? 0.5 : 1);
          mp[i * n + j] += rod * ci * cj + top * rig.links[i].length * rig.links[j].length;
        }
      }
    }
    const bw = rig.drive.velocityBandwidth * knobs.driveBandwidthScale;
    this.kp = this.totalMass * bw;
    this.ki = this.kp * bw / 4;
    this.friction[0] = rig.cart.friction;
    this.damping[0] = rig.cart.viscous;
    for (let i = 1; i <= n; i++) {
      this.friction[i] = rig.joints.coulomb * knobs.frictionScale;
      this.damping[i] = rig.joints.viscous * knobs.frictionScale;
    }
    // MuJoCo's default constraint impedance uses the inverse inertia at qpos0.
    // Preserve the live pose while computing this constant upright inertia.
    this.scratch.set(this.q);
    this.q.fill(0);
    this.buildMass();
    this.invertMass();
    for (let i = 0; i <= n; i++) this.regularization[i] = this.inverse[i * (n + 1) + i] / 9;
    this.q.set(this.scratch);
    this.updateAngles();
  }

  reset(theta?: ArrayLike<number>, cartX = 0): void {
    if (!Number.isFinite(cartX) || (theta && (theta.length !== this.n || Array.from(theta).some(v => !Number.isFinite(v))))) {
      throw new Error('reset requires finite cart position and one absolute angle per link');
    }
    this.q[0] = cartX;
    for (let i = 0; i < this.n; i++) this.q[i + 1] = (theta?.[i] ?? 0) - (i ? theta?.[i - 1] ?? 0 : 0);
    this.velocity.fill(0);
    this.frictionForce.fill(0);
    this.driveOn = true;
    this.t = this.force = this.a = this.setpoint = this.integral = 0;
    this.pushes.length = 0;
    this.hitRailEnd = Math.abs(cartX) > this.rig.cart.railHalfLength;
    this.evaluateAcceleration(0, 0);
  }

  private updateAngles(): void {
    let angle = 0, rate = 0;
    for (let i = 0; i < this.n; i++) {
      this.theta[i] = angle += this.q[i + 1];
      this.omega[i] = rate += this.velocity[i + 1];
      this.sin[i] = Math.sin(angle); this.cos[i] = Math.cos(angle);
    }
  }

  private buildMass(): void {
    this.updateAngles();
    const { n, mass: m, mp, inertia, moment, sin, cos } = this, d = n + 1;
    m[0] = this.totalMass;
    // Absolute-angle mass matrix, then T^-T M T^-1 (suffix sums).
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      m[(i + 1) * d + j + 1] = mp[i * n + j] * (cos[i] * cos[j] + sin[i] * sin[j]) + (i === j ? inertia[i] : 0);
    }
    for (let i = n - 1; i >= 0; i--) for (let j = n - 1; j >= 0; j--) {
      const row = i + 1, col = j + 1;
      if (i + 1 < n) m[row * d + col] += m[(row + 1) * d + col];
      if (j + 1 < n) m[row * d + col] += m[row * d + col + 1];
      if (i + 1 < n && j + 1 < n) m[row * d + col] -= m[(row + 1) * d + col + 1];
    }
    let coupling = 0;
    for (let i = n - 1; i >= 0; i--) {
      coupling += moment[i] * cos[i];
      m[i + 1] = m[(i + 1) * d] = coupling;
    }
  }

  private invertMass(): void {
    const d = this.n + 1;
    factor(this.mass, this.chol, d);
    for (let j = 0; j < d; j++) {
      this.unit.fill(0); this.unit[j] = 1;
      solve(this.chol, this.unit, this.acceleration, d);
      for (let i = 0; i < d; i++) this.inverse[i * d + j] = this.acceleration[i];
    }
  }

  push(link: number, force: number, duration = 0.02): void {
    if (!Number.isInteger(link) || link < 0 || link >= this.n) throw new RangeError('Invalid link');
    if (!Number.isFinite(force) || !Number.isFinite(duration) || duration <= 0) throw new RangeError('Invalid push');
    this.pushes.push({ link, force, end: this.t + duration });
  }

  measure(): Measurement {
    const { sensors } = this.rig, k = this.knobs, m = this.measurement;
    const quantum = 2 * Math.PI / 2 ** k.encoderBits;
    m.t = this.t;
    m.cartPosition = roundEven((this.q[0] + this.rng.normal() * sensors.cartNoise * k.sensorNoiseScale) / sensors.cartResolution) * sensors.cartResolution;
    for (let i = 0; i < this.n; i++) {
      m.jointAngles[i] = roundEven((this.q[i + 1] + this.rng.normal() * sensors.encoderNoise * k.sensorNoiseScale) / quantum) * quantum;
    }
    return m;
  }

  step(command: number): void {
    if (!Number.isFinite(command)) throw new RangeError('Command must be finite');
    const { rig, velocity: v, acceleration: acc } = this;
    const dt = rig.timing.physicsDt, d = this.n + 1;
    const accelCommand = clamp(command, this.knobs.accelLimit);
    for (let sub = 0; sub < this.substeps; sub++) {
      if (this.driveOn) {
        this.setpoint = clamp(this.setpoint + accelCommand * dt, rig.drive.velocityLimit);
        const error = this.setpoint - v[0], candidate = this.integral + error * dt;
        const candidateForce = this.kp * error + this.ki * candidate;
        if (Math.abs(candidateForce) <= rig.drive.forceLimit || candidateForce * error <= 0) this.integral = candidate;
        this.force = clamp(this.kp * error + this.ki * this.integral, rig.drive.forceLimit);
      } else this.force = 0;
      this.evaluateAcceleration(dt, dt);
      for (let i = 0; i < d; i++) {
        v[i] += dt * acc[i];
        this.q[i] += dt * v[i];
      }
      if (Math.abs(this.q[0]) > rig.cart.railHalfLength) this.hitRailEnd = true;
      const stop = rig.cart.railHalfLength + 0.05;
      if (Math.abs(this.q[0]) > stop) {
        this.q[0] = clamp(this.q[0], stop);
        if (this.q[0] * v[0] > 0) v[0] = 0;
      }
      this.t += dt;
    }
    // Like mj_forward after mj_step: report acceleration at the new pose,
    // and remove expired pushes before the next sensor sample.
    this.evaluateAcceleration(0, 0);
  }

  private evaluateAcceleration(implicitDt: number, pushDt: number): void {
    const { rig, n, rhs, velocity: v, frictionForce: f, inverse: inv, acceleration: acc } = this;
    const d = n + 1;
    this.buildMass();
    rhs[0] = this.force;
    for (let i = 0; i < n; i++) {
      rhs[0] += this.moment[i] * this.sin[i] * this.omega[i] ** 2;
      let torque = rig.gravity * this.moment[i] * this.sin[i];
      for (let j = 0; j < n; j++) {
        torque -= this.mp[i * n + j] * (this.sin[i] * this.cos[j] - this.cos[i] * this.sin[j]) * this.omega[j] ** 2;
      }
      rhs[i + 1] = torque;
    }
    for (let p = this.pushes.length - 1; p >= 0; p--) {
      const push = this.pushes[p];
      if (push.end <= this.t + 1e-12) { this.pushes[p] = this.pushes[this.pushes.length - 1]; this.pushes.pop(); continue; }
      const force = push.force * (pushDt > 0 ? Math.min(1, (push.end - this.t) / pushDt) : 1);
      rhs[0] += force;
      for (let i = 0; i <= push.link; i++) rhs[i + 1] += force * rig.links[i].length * this.cos[i];
    }
    for (let i = n - 1; i >= 1; i--) rhs[i] += rhs[i + 1];
    for (let i = 0; i < d; i++) rhs[i] -= this.damping[i] * v[i];
    this.invertMass();
    solve(this.chol, rhs, acc, d);
    // Box-constrained convex solve: min .5*f'(Minv+R)*f +
    // f'*(a_free - a_reference), |f_i|<=frictionloss_i. Interior
    // constraints stick (up to MuJoCo's soft impedance); bounds slip.
    for (let iteration = 0; iteration < 80; iteration++) {
      let largest = 0;
      for (let i = 0; i < d; i++) {
        let gradient = acc[i] + FRICTION_VELOCITY_GAIN * v[i] + this.regularization[i] * f[i];
        for (let j = 0; j < d; j++) gradient += inv[i * d + j] * f[j];
        const next = clamp(f[i] - gradient / (inv[i * d + i] + this.regularization[i]), this.friction[i]);
        largest = Math.max(largest, Math.abs(next - f[i]));
        f[i] = next;
      }
      if (largest < 1e-12) break;
    }
    for (let i = 0; i < d; i++) {
      rhs[i] += f[i];
      this.mass[i * d + i] += implicitDt * this.damping[i];
    }
    factor(this.mass, this.chol, d);
    solve(this.chol, rhs, acc, d);
    this.a = acc[0];
  }
}

/** NumPy rint uses ties-to-even, including negative half-counts. */
function roundEven(x: number): number {
  const lower = Math.floor(x), fraction = x - lower;
  return fraction === 0.5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(x);
}
