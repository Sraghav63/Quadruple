import { describe, it, expect, vi } from 'vitest';
import { LQRController, wrapAngle } from '../src/sim/controller';
import { Plant } from '../src/sim/plant';
import { createSimulation, DEFAULT_KNOBS, loadPreset, loadPresetIndex } from '../src/sim';
import type { Preset, RigParams } from '../src/sim/types';
import quad from '../public/presets/quad.json';
import links1 from '../public/presets/links1.json';
import links2 from '../public/presets/links2.json';
import links3 from '../public/presets/links3.json';
import short from '../public/presets/quad-short.json';
import cQuad from './golden/quad.controller.json';
import c1 from './golden/links1.controller.json';
import c2 from './golden/links2.controller.json';
import c3 from './golden/links3.controller.json';
import cShort from './golden/quad-short.controller.json';
import cUUUD from './golden/quad.UUUD.controller.json';
import cDDUU from './golden/quad.DDUU.controller.json';
import pQuad from './golden/quad.plant.json';
import pUUUD from './golden/quad.UUUD.plant.json';
import p1 from './golden/links1.plant.json';

const presets: Preset[] = [quad, links1, links2, links3, short];
const controllers = [cQuad, c1, c2, c3, cShort];

describe('firmware runtime', () => {
  const cases = presets.map((preset, i) => ({ id: preset.id, trace: controllers[i], matrices: preset.controller }));
  for (const trace of [cUUUD, cDDUU]) cases.push({
    id: `quad/${trace.configuration}`, trace, matrices: quad.configurations[trace.configuration as 'UUUD' | 'DDUU'].controller,
  });
  for (const { id, trace, matrices } of cases) it(`matches ${id} Python commands`, () => {
    const controller = new LQRController(matrices);
    const m = { t: 0, cartPosition: trace.cartPosition[0], jointAngles: new Float64Array(trace.jointAngles[0]) };
    controller.reset(m, trace.xRef[0]);
    let error = 0;
    for (let k = 0; k < trace.u.length; k++) {
      m.cartPosition = trace.cartPosition[k]; m.jointAngles.set(trace.jointAngles[k]);
      const u = controller.step(m, trace.xRef[k]);
      error = Math.max(error, Math.abs(u - trace.u[k]) / Math.max(1, Math.abs(trace.u[k])));
    }
    console.log('controller', id, error);
    expect(error).toBeLessThan(1e-9);
  });
  it('wraps with Python endpoint conventions', () => {
    expect(wrapAngle(Math.PI)).toBe(-Math.PI);
    expect(wrapAngle(-Math.PI)).toBe(-Math.PI);
    expect(wrapAngle(3 * Math.PI)).toBe(-Math.PI);
    expect(wrapAngle(-3 * Math.PI)).toBe(-Math.PI);
    for (const x of [-Math.PI - 1e-8, -Math.PI + 1e-8, Math.PI - 1e-8, Math.PI + 1e-8, 0.002]) {
      expect(wrapAngle(x)).toBeGreaterThanOrEqual(-Math.PI);
      expect(wrapAngle(x)).toBeLessThan(Math.PI);
      expect(wrapAngle(x + 8 * Math.PI)).toBeCloseTo(wrapAngle(x), 13);
    }
  });
  it('resets deviations and gives identical commands for whole encoder turns', () => {
    const matrices = quad.configurations.DDUU.controller;
    const a = new LQRController(matrices), b = new LQRController(matrices);
    const m = { t: 0, cartPosition: 0.01, jointAngles: new Float64Array([Math.PI + 0.001, -0.002, -Math.PI + 0.003, -0.001]) };
    const turned = { ...m, jointAngles: new Float64Array(m.jointAngles.map((v, i) => v + (i % 2 ? -4 : 2) * Math.PI)) };
    a.reset(m); b.reset(turned);
    for (let i = 0; i < 4; i++) expect(a.z[matrices.thetaIndex + i]).toBeCloseTo([0.001, -0.001, 0.002, 0.001][i], 13);
    for (let k = 0; k < 100; k++) expect(a.step(m)).toBeCloseTo(b.step(turned), 8);
    for (let i = 0; i < matrices.nz; i++) expect(a.z[i]).toBeCloseTo(b.z[i], 9);
  });
});

describe('MuJoCo open-loop parity', () => {
  for (const [id, data] of [['quad', pQuad], ['links1', p1], ['quad/UUUD', pUUUD]] as const) {
    for (const mode of ['frictionless', 'nominal'] as const) it(`${id} ${mode}`, () => {
      const trace = data[mode];
      const rig: RigParams = trace.rig;
      const plant = new Plant(rig, DEFAULT_KNOBS({ ...quad, rig }));
      plant.reset(trace.theta0);
      let thetaError = 0, xError = 0, peak = 0, forceError = 0, rateError = 0;
      for (let k = 0; k < trace.accelCommand.length; k++) {
        plant.step(trace.accelCommand[k]);
        const expected = trace.states[k];
        xError = Math.max(xError, Math.abs(plant.q[0] - expected.x));
        forceError = Math.max(forceError, Math.abs(plant.force - expected.force));
        rateError = Math.max(rateError, Math.abs(plant.velocity[0] - expected.v));
        for (let i = 0; i < plant.n; i++) {
          thetaError = Math.max(thetaError, Math.abs(plant.theta[i] - expected.theta[i]));
          peak = Math.max(peak, Math.abs(expected.theta[i]));
          rateError = Math.max(rateError, Math.abs(plant.omega[i] - expected.omega[i]));
        }
      }
      console.log('plant', id, mode, { thetaError, xError, forceError, peak, seconds: trace.states.length / rig.timing.controlRate });
      expect(thetaError).toBeLessThan(Math.max(1e-4, peak * 0.005));
      expect(xError).toBeLessThan(0.001);
      // Current implementation matches implicitfast essentially to roundoff.
      expect(thetaError).toBeLessThan(1e-10);
      expect(xError).toBeLessThan(1e-11);
      expect(rateError).toBeLessThan(1e-9);
      expect(forceError).toBeLessThan(1e-9);
    });
  }
});

function run(sim: ReturnType<typeof createSimulation>, steps: number): void {
  for (let i = 0; i < steps; i++) sim.step();
}

describe('equilibrium configurations', () => {
  for (const preset of presets) {
    it(`${preset.id} exports every configuration with an all-up default`, () => {
      const n = preset.rig.links.length;
      expect(Object.keys(preset.configurations)).toHaveLength(2 ** n);
      expect(Object.keys(preset.configurations)[0]).toBe('U'.repeat(n));
      expect(preset.controller).toEqual(preset.configurations['U'.repeat(n)].controller);
      for (const [cfg, data] of Object.entries(preset.configurations)) {
        expect(cfg).toMatch(new RegExp(`^[UD]{${n}}$`));
        expect(data.controller.configuration).toBe(cfg);
        expect(data.controller.thetaEq).toEqual([...cfg].map(c => c === 'U' ? 0 : Math.PI));
        expect(data.controller.closedLoopRadius).toBeLessThan(1);
        expect(data.controller.observerRadius).toBeLessThan(1);
      }
      expect(preset.configurations['D'.repeat(n)].unstablePoles).toEqual([]);
    });
    for (const cfg of Object.keys(preset.configurations)) it(`${preset.id}/${cfg} survives a 20 cm step and 0.08 N tap from equilibrium`, () => {
      const sim = createSimulation(preset, 0);
      sim.setConfiguration(cfg);
      sim.reset();
      run(sim, 1000); sim.setReference(0.2);
      run(sim, 2000); sim.push(preset.rig.links.length - 1, 0.08, 0.02);
      run(sim, 3000);
      expect(sim.configuration).toBe(cfg);
      expect(sim.status().fallen).toBe(false);
      expect(sim.status().thetaDeviation.every(v => Math.abs(v) < 0.6)).toBe(true);
    });
  }
  it('resets around the selected equilibrium and reports absolute ghost angles', () => {
    const sim = createSimulation(quad);
    sim.setKnobs({ sensorNoiseScale: 0 });
    sim.setConfiguration(' uuud ');
    expect(sim.configuration).toBe('UUUD');
    expect(Array.from(sim.status().theta)).toEqual([0, 0, 0, Math.PI]);
    expect(sim.status().thetaEstimate[3]).toBeCloseTo(Math.PI, 13);
    sim.reset({ randomTiltDeg: 0.05 });
    expect(sim.status().thetaDeviation.every(v => Math.abs(v) <= 0.05 * Math.PI / 180)).toBe(true);
    run(sim, 20); sim.controllerEnabled = false;
    sim.setConfiguration('DDDD');
    expect(sim.status().t).toBe(0);
    expect(sim.status().command).toBe(0);
    expect(sim.status().driveEnabled).toBe(true);
    expect(Array.from(sim.status().theta)).toEqual(Array(4).fill(Math.PI));
    for (const cfg of ['UUU', 'UUUX', '__proto__']) expect(() => sim.setConfiguration(cfg)).toThrow();
    expect(sim.configuration).toBe('DDDD');
  });
  it('detects falls relative to the target across the angle wrap', () => {
    const sim = createSimulation(quad);
    sim.setConfiguration('DDDD');
    sim.reset({ theta: [-Math.PI + 0.001, Math.PI, -Math.PI, 3 * Math.PI] });
    sim.step();
    expect(sim.status().fallen).toBe(false);
    expect(sim.status().thetaDeviation.every(v => Math.abs(v) < 0.01)).toBe(true);
    sim.reset({ theta: [Math.PI, Math.PI + 0.61, Math.PI, Math.PI] });
    sim.step();
    expect(sim.status().fallen).toBe(true);
    expect(sim.status().driveEnabled).toBe(false);
    sim.setConfiguration('UUUD');
    expect(sim.status().fallen).toBe(false);
  });
});

describe('closed loop', () => {
  it('survives ten seconds from 0.05 degree tilt', () => {
    const sim = createSimulation(quad, 0);
    sim.reset({ theta: Array(4).fill(0.05 * Math.PI / 180) });
    run(sim, 10000);
    expect(sim.status().fallen).toBe(false);
  });
  it('recovers from a 0.08 N, 20 ms top tap', () => {
    const sim = createSimulation(quad, 0);
    sim.reset({ randomTiltDeg: 0.05 }); run(sim, 2000);
    sim.push(3, 0.08, 0.02); run(sim, 10000);
    expect(sim.status().fallen).toBe(false);
  });
  it('falls when disarmed and continues finite physics with zero commands', () => {
    const sim = createSimulation(quad);
    sim.reset({ theta: [0.01, 0.01, 0.01, 0.01] }); sim.controllerEnabled = false;
    run(sim, 10000);
    expect(sim.status().fallen).toBe(true);
    expect(sim.status().command).toBe(0);
    expect(sim.status().force).toBe(0);
    expect(sim.status().vSetpoint).toBe(0);
    expect(sim.status().driveEnabled).toBe(false);
    expect(sim.status().theta.every(Number.isFinite)).toBe(true);
    expect(Math.abs(sim.status().x)).toBeLessThanOrEqual(quad.rig.cart.railHalfLength + 0.05);
    expect(sim.status().t).toBeCloseTo(10, 8);
  });
  it('tracks a 0.2 m reference', () => {
    const sim = createSimulation(quad, 0);
    sim.reset({ randomTiltDeg: 0.05 }); run(sim, 1000); sim.setReference(0.2);
    run(sim, 7000);
    let mean = 0;
    for (let i = 0; i < 3000; i++) { sim.step(); mean += sim.status().x / 3000; }
    console.log('tracking mean', mean, 'fallen', sim.status().fallen);
    expect(sim.status().fallen).toBe(false);
    expect(Math.abs(mean - 0.2)).toBeLessThan(0.08);
  });
  it('runs 1000 quad control periods within 50 ms', () => {
    const sim = createSimulation(quad, 7); run(sim, 2000);
    const times: number[] = [];
    for (let k = 0; k < 5; k++) {
      const start = performance.now(); run(sim, 1000); times.push(performance.now() - start);
    }
    times.sort((a, b) => a - b);
    console.log('1000-step timings ms', times);
    expect(times[4]).toBeLessThan(50);
    expect(sim.status().fallen).toBe(false);
  });
  it('is deterministic with the same seed and inputs', () => {
    const a = createSimulation(quad, 42), b = createSimulation(quad, 42), c = createSimulation(quad, 43);
    a.reset({ randomTiltDeg: 0.05 }); b.reset({ randomTiltDeg: 0.05 }); c.reset({ randomTiltDeg: 0.05 });
    a.push(2, 0.02); b.push(2, 0.02);
    run(a, 1000); run(b, 1000); run(c, 1000);
    expect(a.status()).toEqual(b.status());
    expect(a.status().theta).not.toEqual(c.status().theta);
  });
  it('carries fractional time and step-capped backlog', () => {
    const sim = createSimulation(quad);
    expect(sim.advance(0.0004)).toBe(0);
    expect(sim.advance(0.0004)).toBe(0);
    expect(sim.advance(0.0004)).toBe(1);
    expect(sim.advance(0.2, 10)).toBe(10);
    expect(sim.advance(0, 200)).toBe(190);
    expect(sim.status().t).toBeCloseTo(0.201, 10);
  });
  it('changes true plant knobs without changing controller matrices', () => {
    const sim = createSimulation(quad);
    const matrices = JSON.stringify(quad.controller);
    sim.setKnobs({ massScale: 1.1, frictionScale: 1.2, extraDelaySteps: 3, encoderBits: 16, sensorNoiseScale: 0 });
    run(sim, 20);
    expect(JSON.stringify(quad.controller)).toBe(matrices);
    expect(sim.knobs.massScale).toBe(1.1);
    expect(() => sim.setKnobs({ massScale: -1 })).toThrow();
    expect(sim.knobs.massScale).toBe(1.1);
  });
});

describe('preset loading under a subpath', () => {
  it('respects root-relative and absolute deployment bases', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(quad) });
    vi.stubGlobal('fetch', fetchMock);
    try {
      await loadPresetIndex('/pendulum/'); await loadPreset('quad', 'https://example.org/demo');
      expect(fetchMock.mock.calls[0][0]).toBe('/pendulum/presets/index.json');
      expect(fetchMock.mock.calls[1][0]).toBe('https://example.org/demo/presets/quad.json');
    } finally { vi.unstubAllGlobals(); }
  });
});

describe('sensors, disturbances and safety', () => {
  it('disables a moving drive immediately and re-enables at measured speed', () => {
    const rig = { ...links1.rig, gravity: 0, cart: { ...links1.rig.cart, friction: 0, viscous: 0 } };
    const p = new Plant(rig, { ...DEFAULT_KNOBS(links1), frictionScale: 0 });
    for (let i = 0; i < 100; i++) p.step(2);
    expect(p.vSetpoint).toBeGreaterThan(0.1);
    expect(Math.abs(p.force)).toBeGreaterThan(0);
    p.setDriveEnabled(false);
    expect(p.force).toBe(0);
    expect(p.vSetpoint).toBe(0);
    p.q.fill(0); p.velocity.fill(0); p.velocity[0] = 0.2;
    const x = p.q[0];
    for (let i = 0; i < 20; i++) p.step(30); // commands cannot energize a disabled drive
    expect(p.force).toBe(0);
    expect(p.vSetpoint).toBe(0);
    expect(p.velocity[0]).toBeCloseTo(0.2, 12);
    expect(p.q[0]).toBeCloseTo(x + 0.2 * 20 / rig.timing.controlRate, 12);
    p.setDriveEnabled(true);
    expect(p.vSetpoint).toBe(p.velocity[0]);
    p.step(0);
    expect(p.force).toBeCloseTo(0, 12); // no retained PI torque
    p.setDriveEnabled(false);
    p.reset();
    expect(p.driveEnabled).toBe(true);
    expect(p.vSetpoint).toBe(0);
  });
  it('keeps rail friction and end stops active with the drive off', () => {
    const p = new Plant(links1.rig, DEFAULT_KNOBS(links1));
    p.velocity[0] = 0.2;
    p.setDriveEnabled(false);
    for (let i = 0; i < 20; i++) p.step(0);
    expect(p.velocity[0]).toBeGreaterThan(0);
    expect(p.velocity[0]).toBeLessThan(0.2);
    expect(p.force).toBe(0);
    p.reset(undefined, links1.rig.cart.railHalfLength - 0.001);
    p.velocity[0] = 1;
    p.setDriveEnabled(false);
    for (let i = 0; i < 200; i++) p.step(30);
    expect(p.hitRailEnd).toBe(true);
    expect(p.q[0]).toBeLessThanOrEqual(links1.rig.cart.railHalfLength + 0.05);
    expect(p.force).toBe(0);
  });
  it('flushes queued commands even when toggled off and back on between steps', () => {
    const sim = createSimulation(links1);
    sim.setKnobs({ extraDelaySteps: 3, sensorNoiseScale: 0 });
    sim.setReference(0.2);
    sim.step();
    sim.controllerEnabled = false;
    expect(sim.status().driveEnabled).toBe(false);
    sim.controllerEnabled = true;
    expect(sim.status().vSetpoint).toBe(sim.status().v);
    for (let i = 0; i < links1.rig.timing.computeDelaySteps + 3; i++) {
      sim.step();
      expect(sim.status().command).toBe(0);
    }
    sim.step();
    expect(Math.abs(sim.status().command)).toBeGreaterThan(0);
    expect(sim.status().driveEnabled).toBe(true);
  });
  it('controller off clears a live setpoint and reset re-arms both controller and drive', () => {
    const sim = createSimulation(links1);
    sim.setReference(0.1); run(sim, 50);
    expect(Math.abs(sim.status().vSetpoint)).toBeGreaterThan(0);
    sim.controllerEnabled = false;
    expect(sim.status().force).toBe(0);
    expect(sim.status().vSetpoint).toBe(0);
    run(sim, 20);
    expect(sim.status().force).toBe(0);
    expect(sim.status().driveEnabled).toBe(false);
    sim.reset();
    expect(sim.status().fallen).toBe(false);
    expect(sim.controllerEnabled).toBe(true);
    expect(sim.status().driveEnabled).toBe(true);
    run(sim, 50);
    expect(Math.abs(sim.status().force)).toBeGreaterThan(0);
  });
  it('clears a live velocity setpoint on the exact step a push causes a fall', () => {
    const sim = createSimulation(links1);
    sim.setReference(0.1); run(sim, 50);
    expect(Math.abs(sim.status().vSetpoint)).toBeGreaterThan(0);
    sim.push(0, 20, 0.02);
    let previousSetpoint = 0;
    for (let i = 0; i < 2000 && !sim.status().fallen; i++) {
      previousSetpoint = sim.status().vSetpoint;
      sim.step();
    }
    expect(sim.status().fallen).toBe(true);
    expect(Math.abs(previousSetpoint)).toBeGreaterThan(0);
    expect(sim.status().command).toBe(0);
    expect(sim.status().force).toBe(0);
    expect(sim.status().vSetpoint).toBe(0);
    expect(sim.status().driveEnabled).toBe(false);
  });
  it('quantizes relative encoders after noise with NumPy ties-to-even', () => {
    const knobs = DEFAULT_KNOBS(quad);
    knobs.sensorNoiseScale = 0;
    const p = new Plant(quad.rig, knobs);
    const quantum = 2 * Math.PI / 2 ** knobs.encoderBits;
    p.q.set([2.5 * quad.rig.sensors.cartResolution, 0.5 * quantum, 1.5 * quantum, -0.5 * quantum, -1.5 * quantum]);
    const m = p.measure();
    expect(m.cartPosition).toBe(2 * quad.rig.sensors.cartResolution);
    expect(Array.from(m.jointAngles)).toEqual([0, 2 * quantum, 0, -2 * quantum]);
  });
  it('adds overlapping pushes and expires them at the physics boundary', () => {
    const a = new Plant(quad.rig, DEFAULT_KNOBS(quad)), b = new Plant(quad.rig, DEFAULT_KNOBS(quad));
    a.push(3, 0.08, 0.0201); b.push(3, 0.04, 0.0201); b.push(3, 0.04, 0.0201);
    for (let i = 0; i < 100; i++) { a.step(0); b.step(0); }
    for (let i = 0; i < a.q.length; i++) expect(a.q[i]).toBeCloseTo(b.q[i], 11);
    expect(Math.abs(a.theta[3])).toBeGreaterThan(1e-4);
    expect(() => a.push(4, 1)).toThrow();
    expect(() => a.push(0, 1, 0)).toThrow();
  });
  it('stops a driven cart at the physical end stop', () => {
    const p = new Plant(links1.rig, DEFAULT_KNOBS(links1));
    for (let i = 0; i < 2000; i++) p.step(30);
    expect(p.hitRailEnd).toBe(true);
    expect(p.q[0]).toBeCloseTo(links1.rig.cart.railHalfLength + 0.05, 10);
    expect(p.velocity[0]).toBe(0);
    expect(p.theta.every(Number.isFinite)).toBe(true);
  });
  it('disarms automatically on a fall while continuing physics', () => {
    const sim = createSimulation(quad);
    sim.reset({ theta: [0.7, 0.7, 0.7, 0.7] }); sim.step();
    expect(sim.status().controllerEnabled).toBe(true);
    expect(sim.status().fallen).toBe(true);
    expect(sim.status().command).toBe(0);
    expect(sim.status().force).toBe(0);
    expect(sim.status().vSetpoint).toBe(0);
    expect(sim.status().driveEnabled).toBe(false);
    sim.controllerEnabled = false; sim.controllerEnabled = true;
    expect(sim.status().driveEnabled).toBe(false); // a fall stays latched until reset
    const theta = Array.from(sim.status().theta);
    run(sim, 100);
    expect(Array.from(sim.status().theta)).not.toEqual(theta);
    expect(sim.status().command).toBe(0);
    expect(sim.status().force).toBe(0);
    expect(sim.status().vSetpoint).toBe(0);
    sim.reset();
    expect(sim.status().fallen).toBe(false);
    expect(sim.status().driveEnabled).toBe(true);
  });
  it('preserves the live pose during a true-mass update and clips reference', () => {
    const sim = createSimulation(quad, 11);
    sim.reset({ theta: [0.001, 0.002, 0.003, 0.004] }); run(sim, 10);
    const theta = Array.from(sim.status().theta), x = sim.status().x;
    sim.setKnobs({ massScale: 1.2 });
    expect(Array.from(sim.status().theta)).toEqual(theta);
    expect(sim.status().x).toBe(x);
    sim.setReference(5);
    expect(sim.status().xRef).toBe(0.8 * quad.rig.cart.railHalfLength);
  });
});
