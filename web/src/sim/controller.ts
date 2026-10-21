import type { ControllerMatrices, Measurement } from './types';

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
      z[m.thetaIndex + i - 1] = theta;
    }
    this.xRef = xRef;
  }

  step(measurement: Measurement, xRef = 0): number {
    const { z, matrices: m, innovation, next } = this;
    z[0] -= xRef - this.xRef;
    this.xRef = xRef;
    let theta = 0;
    for (let i = 0; i < m.ny; i++) {
      if (i > 0) theta += measurement.jointAngles[i - 1];
      let predicted = 0;
      for (let j = 0; j < m.nz; j++) predicted += m.C[i][j] * z[j];
      innovation[i] = (i === 0 ? measurement.cartPosition - xRef : theta) - predicted;
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
