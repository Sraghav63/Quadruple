import type { RigParams } from './types';
import { zeros, solve, type Matrix } from './matrix';

export class ChainDynamics {
  readonly mp: Matrix;
  readonly inertia: Float64Array;
  readonly b: Float64Array;
  readonly damping: Matrix;
  constructor(readonly rig: RigParams) {
    const n = rig.links.length;
    this.mp = zeros(n); this.inertia = new Float64Array(n); this.b = new Float64Array(n); this.damping = zeros(n);
    rig.links.forEach((link, i) => {
      this.inertia[i] = link.rodMass * link.length ** 2 / 12;
      for (const [mass, own] of [[link.rodMass, link.length / 2], [link.topMass, link.length]]) {
        const c = rig.links.map((l, j) => j < i ? l.length : j === i ? own : 0);
        for (let j = 0; j < n; j++) {
          this.b[j] += mass * c[j];
          for (let k = 0; k < n; k++) this.mp[j][k] += mass * c[j] * c[k];
        }
      }
      this.damping[i][i] += rig.joints.viscous;
      if (i > 0) {
        this.damping[i - 1][i - 1] += rig.joints.viscous;
        this.damping[i][i - 1] -= rig.joints.viscous;
        this.damping[i - 1][i] -= rig.joints.viscous;
      }
    });
  }
  angularAcceleration(theta: ArrayLike<number>, omega: ArrayLike<number>, accel: number): Float64Array {
    const n = theta.length, m = zeros(n), rhs = zeros(n, 1);
    for (let i = 0; i < n; i++) {
      rhs[i][0] = this.rig.gravity * this.b[i] * Math.sin(theta[i]) - this.b[i] * Math.cos(theta[i]) * accel;
      for (let j = 0; j < n; j++) {
        const d = theta[i] - theta[j];
        m[i][j] = this.mp[i][j] * Math.cos(d) + (i === j ? this.inertia[i] : 0);
        rhs[i][0] -= this.mp[i][j] * Math.sin(d) * omega[j] ** 2 + this.damping[i][j] * omega[j];
      }
    }
    return new Float64Array(solve(m, rhs).map(row => row[0]));
  }
  jacobian(theta: ArrayLike<number>, omega: ArrayLike<number>, accel: number, step = 1e-6): Matrix {
    const n = theta.length, base = new Float64Array([accel, ...Array.from(theta), ...Array.from(omega)]), jac = zeros(n, 2 * n + 1);
    for (let j = 0; j < base.length; j++) {
      const hi = base.slice(), lo = base.slice(); hi[j] += step; lo[j] -= step;
      const ah = this.angularAcceleration(hi.subarray(1, n + 1), hi.subarray(n + 1), hi[0]);
      const al = this.angularAcceleration(lo.subarray(1, n + 1), lo.subarray(n + 1), lo[0]);
      for (let i = 0; i < n; i++) jac[i][j] = (ah[i] - al[i]) / (2 * step);
    }
    return jac;
  }
}
