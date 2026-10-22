import type { ControllerMatrices, Measurement } from './types';

/** Same modulo convention as Python: [-pi, pi). */
export function wrapAngle(angle: number): number {
  const w = (angle + Math.PI) % (2 * Math.PI);
  return (w < 0 ? w + 2 * Math.PI : w) - Math.PI;
}

/** Fixed matrix firmware runtime. All scratch storage is allocated once. */
export class LQRController {
  readonly z: Float64Array;
  private readonly innovation: Float64Array;
  private readonly next: Float64Array;
  private xRef = 0;

  constructor(readonly matrices: ControllerMatrices) {
    this.z = new Float64Array(matrices.nz);
    this.next = new Float64Array(matrices.nz);
    this.innovation = new Float64Array(matrices.ny);
  }

  reset(measurement: Measurement, xRef = 0): void {
    const { z, matrices: m } = this;
    z.fill(0);
    z[0] = measurement.cartPosition - xRef;
    let theta = 0;
    for (let i = 1; i < m.ny; i++) {
      theta += measurement.jointAngles[i - 1];
      z[m.thetaIndex + i - 1] = wrapAngle(theta - m.thetaEq[i - 1]);
    }
    this.xRef = xRef;
  }

  seed(deviation: ArrayLike<number>, inFlight: ArrayLike<number>, xRef = 0): void {
    const m = this.matrices;
    this.z.fill(0);
    for (let i = 0; i < 3 + 2 * (m.ny - 1); i++) this.z[i] = deviation[i];
    const { delay, nx } = m;
    for (let i = 0; i < delay; i++) this.z[nx + i] = inFlight[i];
    this.xRef = xRef;
  }

  get physicalEstimate(): Float64Array { return this.z.slice(0, 3 + 2 * (this.matrices.ny - 1)); }

  step(measurement: Measurement, xRef = 0): number {
    const { z, matrices: m, innovation, next } = this;
    z[0] -= xRef - this.xRef;
    this.xRef = xRef;
    let theta = 0;
    for (let i = 0; i < m.ny; i++) {
      if (i > 0) theta += measurement.jointAngles[i - 1];
      let predicted = 0;
      for (let j = 0; j < m.nz; j++) predicted += m.C[i][j] * z[j];
      innovation[i] = (i === 0 ? measurement.cartPosition - xRef : wrapAngle(theta - m.thetaEq[i - 1])) - predicted;
    }
    let u = 0;
    for (let i = 0; i < m.nz; i++) {
      let correction = 0;
      for (let j = 0; j < m.ny; j++) correction += m.L[i][j] * innovation[j];
      z[i] += correction;
      u -= m.K[i] * z[i];
    }
    u = Math.max(-m.accelLimit, Math.min(m.accelLimit, u));
    for (let i = 0; i < m.nz; i++) {
      let value = 0;
      for (let j = 0; j < m.nz; j++) value += m.A[i][j] * z[j];
      next[i] = value + m.B[i] * u;
    }
    z.set(next);
    return u;
  }
}
