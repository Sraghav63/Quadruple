import { afterEach, describe, expect, it, vi } from 'vitest';
import { bindControls, type Actions } from '../src/ui/controls';
import { History, PlotCanvas, WINDOW_S } from '../src/ui/plots';
import type { Renderer } from '../src/ui/renderer';
import { parseHash, resolveConfiguration } from '../src/ui/location';
import quad from '../public/presets/quad.json';
import links2 from '../public/presets/links2.json';

afterEach(() => vi.unstubAllGlobals());

describe('preset and configuration hashes', () => {
  const list = [quad, links2];
  it.each(['#quad', 'quad'])('keeps legacy %s links all-up', hash => {
    const parsed = parseHash(hash, list);
    expect(parsed.presetId).toBe('quad');
    expect(resolveConfiguration(quad, parsed.configuration)).toBe('UUUU');
  });
  it.each(['UUUD', 'DDUU', 'DDDD'])('parses #quad/%s', cfg => {
    expect(parseHash(`#quad/${cfg}`, list)).toEqual({ presetId: 'quad', configuration: cfg });
    expect(resolveConfiguration(quad, cfg)).toBe(cfg);
  });
  it('normalizes case, respects link count and falls back for malformed hashes', () => {
    expect(parseHash('#links2/du', list)).toEqual({ presetId: 'links2', configuration: 'DU' });
    expect(resolveConfiguration(links2, 'DU')).toBe('DU');
    expect(resolveConfiguration(links2, 'UUUD')).toBe('UU');
    for (const hash of ['#quad/nope', '#quad/UUUD/extra', '#quad/']) {
      expect(resolveConfiguration(quad, parseHash(hash, list).configuration)).toBe('UUUU');
    }
    for (const hash of ['', '#unknown', '#../nope']) expect(parseHash(hash, list).presetId).toBe('quad');
    expect(resolveConfiguration(quad, '__proto__')).toBe('UUUU');
  });
});

class PointerCanvas extends EventTarget {
  private captured = new Set<number>();
  setPointerCapture(id: number): void { this.captured.add(id); }
  hasPointerCapture(id: number): boolean { return this.captured.has(id); }
  releasePointerCapture(id: number): void { this.captured.delete(id); }

  pointer(type: string, props: Partial<PointerEvent> = {}): void {
    const e = new Event(type, { cancelable: true });
    const values = {
      pointerId: 1, isPrimary: true, pointerType: 'touch', button: 0,
      clientX: 100, clientY: 100, timeStamp: 0, ...props,
    };
    for (const [key, value] of Object.entries(values)) Object.defineProperty(e, key, { value });
    this.dispatchEvent(e);
  }
}

function controls(hit: { link: number; dir: 1 | -1 } | null = { link: 2, dir: -1 }) {
  vi.stubGlobal('window', new EventTarget());
  vi.stubGlobal('document', { getElementById: () => new EventTarget() });
  const canvas = new PointerCanvas();
  const renderer = {
    canvasPoint: (x: number, y: number) => [x - 20, y - 30],
    hitLink: vi.fn(() => hit),
    screenToWorldX: (x: number) => x / 100,
  };
  const act = {
    togglePause: vi.fn(), reset: vi.fn(), toggleController: vi.fn(),
    nudgeTarget: vi.fn(), setTargetX: vi.fn(), push: vi.fn(),
  } satisfies Actions;
  bindControls(canvas as unknown as HTMLCanvasElement, renderer as unknown as Renderer, act);
  return { canvas, renderer, act };
}

describe('scene pointer gestures', () => {
  it.each(['touch', 'mouse', 'pen'])('pushes only after a short %s tap, preserving the hit direction', (pointerType) => {
    const { canvas, renderer, act } = controls();
    canvas.pointer('pointerdown', { pointerType });
    expect(canvas.hasPointerCapture(1)).toBe(true);
    expect(act.push).not.toHaveBeenCalled();
    canvas.pointer('pointermove', { clientX: 104, timeStamp: 100 });
    canvas.pointer('pointerup', { clientX: 104, timeStamp: 200 });
    expect(renderer.hitLink).toHaveBeenCalledWith(80, 70, pointerType === 'mouse' ? 18 : 30);
    expect(act.push).toHaveBeenCalledExactlyOnceWith(2, -1);
    expect(act.setTargetX).not.toHaveBeenCalled();
    expect(canvas.hasPointerCapture(1)).toBe(false);
  });

  it('turns a press starting on a link into a continuous target drag', () => {
    const { canvas, act } = controls();
    canvas.pointer('pointerdown');
    canvas.pointer('pointermove', { clientY: 109, timeStamp: 20 });
    expect(act.setTargetX).toHaveBeenLastCalledWith(0.8);
    canvas.pointer('pointermove', { clientX: 140, timeStamp: 40 });
    expect(act.setTargetX).toHaveBeenLastCalledWith(1.2);
    canvas.pointer('pointerup', { timeStamp: 100 }); // returning to start is still a drag
    expect(act.push).not.toHaveBeenCalled();
    expect(act.setTargetX).toHaveBeenLastCalledWith(0.8);
  });

  it('checks movement on release even without a pointermove event', () => {
    const { canvas, act } = controls();
    canvas.pointer('pointerdown');
    canvas.pointer('pointerup', { clientX: 109, timeStamp: 100 });
    expect(act.push).not.toHaveBeenCalled();
    expect(act.setTargetX).toHaveBeenCalledExactlyOnceWith(0.89);
  });

  it('moves the target immediately when a press starts off the links', () => {
    const { canvas, act } = controls(null);
    canvas.pointer('pointerdown');
    expect(act.setTargetX).toHaveBeenLastCalledWith(0.8);
    canvas.pointer('pointermove', { clientX: 110 });
    expect(act.setTargetX).toHaveBeenLastCalledWith(0.9);
    canvas.pointer('pointerup', { timeStamp: 100 });
    expect(act.push).not.toHaveBeenCalled();
  });

  it('does not push on a long press', () => {
    const { canvas, act } = controls();
    canvas.pointer('pointerdown');
    canvas.pointer('pointerup', { timeStamp: 350 });
    expect(act.push).not.toHaveBeenCalled();
    expect(act.setTargetX).not.toHaveBeenCalled();
  });

  it.each(['pointercancel', 'lostpointercapture'])('clears a gesture on %s without pushing and accepts the next press', (event) => {
    const { canvas, act } = controls();
    canvas.pointer('pointerdown');
    canvas.pointer(event);
    canvas.pointer('pointermove', { clientX: 150 });
    canvas.pointer('pointerup', { timeStamp: 100 });
    expect(act.push).not.toHaveBeenCalled();
    expect(act.setTargetX).not.toHaveBeenCalled();
    expect(canvas.hasPointerCapture(1)).toBe(false);
    canvas.pointer('pointerdown', { timeStamp: 200 });
    canvas.pointer('pointerup', { timeStamp: 250 });
    expect(act.push).toHaveBeenCalledTimes(1);
  });

  it('ignores secondary pointers without interrupting the active gesture', () => {
    const { canvas, act } = controls();
    canvas.pointer('pointerdown', { pointerId: 2, isPrimary: false });
    expect(canvas.hasPointerCapture(2)).toBe(false);
    canvas.pointer('pointerdown');
    canvas.pointer('pointerdown', { pointerId: 2, isPrimary: false });
    canvas.pointer('pointermove', { pointerId: 2, clientX: 200 });
    canvas.pointer('pointerup', { pointerId: 2, timeStamp: 50 });
    canvas.pointer('pointercancel', { pointerId: 2 });
    expect(canvas.hasPointerCapture(1)).toBe(true);
    expect(act.setTargetX).not.toHaveBeenCalled();
    canvas.pointer('pointerup', { timeStamp: 100 });
    expect(act.push).toHaveBeenCalledExactlyOnceWith(2, -1);
  });
});

const fillTime = (t: number) => (out: Float32Array, offset: number) => { out[offset] = t; };

describe('simulation-time plot history', () => {
  it.each([[0.5, 120], [1, 60], [2, 60], [2, 24]])('retains ten simulation seconds at speed %s and %s fps', (speed, fps) => {
    const hist = new History(1);
    const end = 30;
    for (let frame = 0; frame <= end * fps / speed; frame++) {
      const t = frame * speed / fps;
      hist.maybeSample(t, fillTime(t));
    }
    expect(hist.time).toBeCloseTo(end, 10);
    expect(hist.timeAt(0)).toBeGreaterThanOrEqual(end - WINDOW_S);
    expect(hist.timeAt(0)).toBeLessThan(end - WINDOW_S + 0.1);
    expect(hist.timeAt(hist.length - 1)).toBeCloseTo(end, 10);
    expect(hist.at(0, 0)).toBeCloseTo(hist.timeAt(0), 5);
  });

  it('expires samples on unsampled frames and resets when sim time goes backwards', () => {
    const hist = new History(1);
    hist.maybeSample(0, fillTime(1));
    hist.maybeSample(10, fillTime(2));
    hist.maybeSample(10.001, fillTime(3)); // too soon to sample, but time still advances
    expect(hist.time).toBe(10.001);
    expect(hist.length).toBe(1);
    expect(hist.at(0, 0)).toBe(2);
    hist.maybeSample(10.0005, fillTime(4)); // reset even though later than last recorded sample
    expect(hist.length).toBe(1);
    expect(hist.timeAt(0)).toBe(10.0005);
    expect(hist.at(0, 0)).toBe(4);
    hist.maybeSample(10.0005, fillTime(5)); // paused
    expect(hist.length).toBe(1);
    expect(hist.at(0, 0)).toBe(4);
    hist.clear();
    expect(hist.length).toBe(0);
    expect(hist.time).toBe(0);
  });

  it('reuses channel storage as the ring wraps', () => {
    const hist = new History(2);
    let storage: Float32Array | undefined;
    let t = 0;
    const fill = (out: Float32Array, offset: number) => {
      storage ??= out;
      expect(out).toBe(storage);
      out[offset] = t; out[offset + 1] = -t;
    };
    for (let i = 0; i <= 2000; i++) { t = i * 0.02; hist.maybeSample(t, fill); }
    expect(hist.timeAt(0)).toBeCloseTo(30, 10);
    for (let i = 0; i < hist.length; i++) {
      expect(hist.at(i, 0)).toBeCloseTo(hist.timeAt(i), 5);
      expect(hist.at(i, 1)).toBeCloseTo(-hist.timeAt(i), 5);
    }
  });

  it('positions irregularly spaced samples by timestamp against the current sim time', () => {
    const ctx = {
      setTransform: vi.fn(), clearRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(),
      stroke: vi.fn(), fillText: vi.fn(), save: vi.fn(), rect: vi.fn(), clip: vi.fn(),
      setLineDash: vi.fn(), restore: vi.fn(),
    };
    vi.stubGlobal('window', { devicePixelRatio: 1 });
    vi.stubGlobal('ResizeObserver', class { observe(): void {} });
    const canvas = {
      getContext: () => ctx, getBoundingClientRect: () => ({ width: 246, height: 116 }),
      width: 0, height: 0,
    };
    const plot = new PlotCanvas(canvas as unknown as HTMLCanvasElement, { unit: 'm', scale: 1, symmetric: true });
    plot.series = [{ channel: 0, color: '#fff' }];
    plot.resize();
    const hist = new History(1);
    for (const t of [9.5, 10.1, 12.4]) hist.maybeSample(t, fillTime(0));
    hist.maybeSample(12.401, fillTime(0)); // latest sample is slightly before "now"
    plot.draw(hist);
    expect(ctx.moveTo.mock.calls.at(-1)![0]).toBeCloseTo(38 + 200 * (1 - 2.901 / 10), 9);
    expect(ctx.lineTo.mock.calls.at(-2)![0]).toBeCloseTo(38 + 200 * (1 - 2.301 / 10), 9);
    expect(ctx.lineTo.mock.calls.at(-1)![0]).toBeCloseTo(38 + 200 * (1 - 0.001 / 10), 9);
  });
});
