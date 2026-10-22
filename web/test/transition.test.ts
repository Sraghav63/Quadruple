import { describe, it, expect } from 'vitest';
import { Trajectory, TransitionController, TransitionLibrary, type TrajectoryData, type TransitionEntry } from '../src/sim/transition';
import { dareDoubling, expm, mul, add, transpose, zeros, maxAbs } from '../src/sim/matrix';
import { createSimulation } from '../src/sim';
import type { Preset } from '../src/sim';
import links1 from '../public/presets/links1.json';
import links2 from '../public/presets/links2.json';
import quad from '../public/presets/quad.json';
import golden from './golden/links2.transition.json';

const trajectories = import.meta.glob<TrajectoryData>('../public/presets/*/transitions/*-*.json', { eager: true, import: 'default' });
const sourceQuad = import.meta.glob('../../data/transitions/quad/DDDD-UUUU.json');
const load = (id: string, file: string): TrajectoryData => {
  const data = trajectories[`../public/presets/${id}/transitions/${file}`];
  if (!data) throw new Error(`Missing exported trajectory ${id}/${file}; rerun scripts/export_web.py`);
  return data;
};
const design = (preset: Preset, file: string) => {
  const ctl = new TransitionController(preset.rig, new Trajectory(load(preset.id, file)));
  const start = performance.now(); for (const _ of ctl.design()) { /* synchronous test driver */ }
  console.log('design', preset.id, performance.now() - start, 'ms'); return ctl;
};
const relativeError = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let err = 0; for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i]))); return err;
};

describe('transition numerical parity', () => {
  it('matches Python samples, ZOH, Riccati gains, online Kalman states and commands', () => {
    const ctl = design(links2, 'DD-UU.json');
    let gainError = 0, matrixError = 0, sampleError = 0;
    for (const [key, expected] of Object.entries(golden.gains)) {
      const k = Number(key); gainError = Math.max(gainError, relativeError(ctl.gains[k], expected));
      sampleError = Math.max(sampleError, relativeError(ctl.nominal.map(row => row[k]), golden.nominal[key as keyof typeof golden.nominal]));
      const a = golden.az[key as keyof typeof golden.az];
      for (let i = 0; i < ctl.nz; i++) matrixError = Math.max(matrixError, relativeError(ctl.az[k][i], a[i]));
    }
    ctl.begin(golden.deviation, golden.inFlight, golden.xOffset);
    let commandError = 0, stateError = 0;
    for (let k = 0; k < golden.u.length; k++) {
      const m = golden.measurements[k];
      const u = ctl.step({ t: k * ctl.dt, cartPosition: m.cartPosition, jointAngles: new Float64Array(m.jointAngles) });
      commandError = Math.max(commandError, Math.abs(u - golden.u[k]));
      const expected = (golden.z as Record<string, number[]>)[k];
      if (expected) stateError = Math.max(stateError, relativeError(ctl.z, expected));
    }
    console.log({ sampleError, matrixError, gainError, commandError, stateError });
    expect(sampleError).toBeLessThan(1e-12); expect(matrixError).toBeLessThan(1e-10);
    expect(gainError).toBeLessThan(1e-7); expect(stateError).toBeLessThan(2e-9); expect(commandError).toBeLessThan(1e-7);
    expect(ctl.done).toBe(true); expect(ctl.handoff()[0]).toBe(ctl.z[0] + golden.xOffset);
  });
  it('expm matches an analytic rotation, stiff lag and nilpotent ZOH', () => {
    const e = expm([new Float64Array([0, -8]), new Float64Array([8, 0])]);
    expect(e[0][0]).toBeCloseTo(Math.cos(8), 13); expect(e[1][0]).toBeCloseTo(Math.sin(8), 13);
    expect(expm([new Float64Array([-23])])[0][0]).toBeCloseTo(Math.exp(-23), 20);
    const j = zeros(3); j[0][1] = j[1][2] = 0.3;
    const zoh = expm(j); expect(zoh[0][2]).toBeCloseTo(0.045, 14); expect(zoh[0][1]).toBeCloseTo(0.3, 14);
  });
  it('doubling equals the stationary Riccati solution and has a small residual', () => {
    const a = [new Float64Array([1, 0.01]), new Float64Array([0, 1])], b = [new Float64Array([0.00005]), new Float64Array([0.01])];
    const q = [new Float64Array([4, 0]), new Float64Array([0, 1])];
    const p = dareDoubling(a, b, q, [new Float64Array([1])]);
    expect(p[0][0]).toBeCloseTo(449.2203036396065, 6); expect(p[0][1]).toBeCloseTo(200.002499985, 6);
    const at = transpose(a), bt = transpose(b), btp = mul(bt, p), g = mul(btp, a)[0].map(v => v / (1 + mul(btp, b)[0][0]));
    expect(maxAbs(add(add(q, mul(mul(at, p), add(a, mul(b, [g]), -1))), p, -1))).toBeLessThan(1e-9);
  });
});

describe('sparse transition routes', () => {
  it('chooses quickest directed routes and caches lazily loaded files', async () => {
    const entries: TransitionEntry[] = [
      { start: 'DD', goal: 'UU', duration: 9, file: 'direct' },
      { start: 'DD', goal: 'DU', duration: 2, file: 'one' },
      { start: 'DU', goal: 'UU', duration: 3, file: 'two' },
    ];
    let loads = 0;
    const lib = new TransitionLibrary(entries, async () => { loads++; return load('links2', 'DD-UU.json'); });
    expect(lib.route('DD', 'UU')?.map(e => e.file)).toEqual(['one', 'two']);
    expect(lib.route('UU', 'DD')).toBeNull(); expect(lib.route('DU', 'DU')).toEqual([]); expect(loads).toBe(0);
    await Promise.all([lib.get(entries[0]), lib.get(entries[0])]); expect(loads).toBe(1);
  });
});

for (const [preset, goal] of [[links1, 'U'], [links2, 'UU'], [quad, 'UUUU']] as const) {
  const available = preset.id !== 'quad' || Object.keys(sourceQuad).length > 0;
  it.skipIf(!available)(`${preset.id} completes a live swing-up and holds ${goal}`, async () => {
    const lib = new TransitionLibrary(preset.transitions ?? [], async e => load(preset.id, e.file));
    const sim = createSimulation(preset, 0, lib);
    expect(Array.from(sim.status().theta)).toEqual(preset.rig.links.map(() => Math.PI));
    const request = sim.requestConfiguration(goal);
    sim.advance(0.1, 100); expect(sim.status().t).toBeGreaterThan(0); expect(sim.status().mode).toBe('center');
    expect(await request).toBe(true);
    let transitioned = false, peakError = 0;
    for (let i = 0; i < 30000; i++) {
      sim.step(); const st = sim.status(); transitioned ||= st.mode === 'transition';
      peakError = Math.max(peakError, ...Array.from(st.thetaDeviation, Math.abs));
      if (st.fallen || (transitioned && st.mode === 'hold' && st.configuration === goal && st.t > 15)) break;
    }
    console.log('closed-loop', preset.id, { t: sim.status().t, mode: sim.status().mode, peakError, designMs: sim.status().designMs });
    expect(transitioned).toBe(true); expect(sim.status().fallen).toBe(false);
    expect(sim.status().mode).toBe('hold'); expect(sim.configuration).toBe(goal);
    expect(sim.status().thetaDeviation.every(v => Math.abs(v) < 0.15)).toBe(true);
    sim.reset(); expect(sim.configuration).toBe('D'.repeat(preset.rig.links.length)); expect(sim.status().t).toBe(0);
  }, 30000);
}

describe('transition supervision edge cases', () => {
  it('reports no route without resetting time, pose or the holding target', async () => {
    const sim = createSimulation(links1, 0, new TransitionLibrary([], async () => load('links1', 'D-U.json')));
    sim.advance(0.05, 50);
    const t = sim.status().t, theta = Array.from(sim.status().theta);
    expect(await sim.requestConfiguration('U')).toBe(false);
    expect(sim.status().transitionError).toContain('No planned route from D to U');
    expect(sim.status().t).toBe(t); expect(Array.from(sim.status().theta)).toEqual(theta);
    expect(sim.status().mode).toBe('hold'); expect(sim.status().target).toBe('D');
  });
  it('keeps balancing after a fetch failure and can retry', async () => {
    let fail = true;
    const sim = createSimulation(links1, 0, new TransitionLibrary(links1.transitions, async () => {
      if (fail) throw new Error('offline'); return load('links1', 'D-U.json');
    }));
    expect(await sim.requestConfiguration('U')).toBe(false);
    expect(sim.status().transitionError).toContain('offline'); expect(sim.status().mode).toBe('hold');
    fail = false;
    expect(await sim.requestConfiguration('U')).toBe(true); expect(sim.status().mode).toBe('center');
    sim.reset();
  });
  it.each(['reset', 'disable'] as const)('cancels preparation on %s without a late handoff', async action => {
    let deliver!: (data: TrajectoryData) => void;
    const lib = new TransitionLibrary(links1.transitions, () => new Promise(resolve => { deliver = resolve; }));
    const sim = createSimulation(links1, 0, lib), request = sim.requestConfiguration('U');
    expect(sim.status().mode).toBe('center');
    expect(await sim.requestConfiguration('D')).toBe(false);
    sim.setReference(0.2); expect(sim.status().xRef).toBe(0);
    if (action === 'reset') sim.reset(); else sim.controllerEnabled = false;
    deliver(load('links1', 'D-U.json'));
    expect(await request).toBe(false); expect(sim.status().mode).toBe('hold'); expect(sim.status().target).toBe('D');
    sim.advance(0.3, 300); expect(sim.configuration).toBe('D'); expect(sim.status().transitionProgress).toBeNull();
    expect(sim.status().driveEnabled).toBe(action === 'reset');
  });
  it('centres a moving cart and hands off across two edges without resetting the clock', async () => {
    const entries = links2.transitions.filter(e => (e.start === 'DD' && e.goal === 'DU') || (e.start === 'DU' && e.goal === 'UU'));
    const sim = createSimulation(links2, 0, new TransitionLibrary(entries, async e => load('links2', e.file)));
    sim.setReference(0.15); for (let k = 0; k < 4000; k++) sim.step();
    expect(sim.status().x).toBeGreaterThan(0.1);
    const before = sim.status().t;
    const request = sim.requestConfiguration('UU');
    expect(sim.status().route).toEqual(['DD', 'DU', 'UU']);
    // The balancing loop can run while asynchronous preparation is pending.
    for (let k = 0; k < 1000; k++) sim.step();
    expect(await request).toBe(true);
    const reached = new Set<string>(); let lastTime = sim.status().t;
    for (let k = 0; k < 25000; k++) {
      sim.step(); const st = sim.status(); reached.add(st.configuration);
      expect(st.t).toBeGreaterThan(lastTime); lastTime = st.t;
      if (st.fallen || (st.mode === 'hold' && st.configuration === 'UU' && st.t > before + 15)) break;
    }
    expect(sim.status().fallen).toBe(false); expect(sim.status().mode).toBe('hold'); expect(sim.configuration).toBe('UU');
    expect(reached.has('DU')).toBe(true); expect(sim.status().xRef).toBe(0);
  }, 30000);
});
