import { THEME, linkColor } from "./colors";

export const SAMPLE_HZ = 60;
export const WINDOW_S = 10;
const CAP = SAMPLE_HZ * WINDOW_S + 1;

/** Fixed-size ring buffer of per-sample channel values; sampled in sim time so pause/slow-mo behave. */
export class History {
  readonly channels: number;
  private data: Float32Array;
  private readonly times = new Float64Array(CAP);
  private head = 0; // next write index
  private count = 0;
  private lastT = -Infinity;
  private now = 0;

  constructor(channels: number) {
    this.channels = channels;
    this.data = new Float32Array(CAP * channels);
  }

  clear(): void { this.head = 0; this.count = 0; this.lastT = -Infinity; this.now = 0; }
  get length(): number { return this.count; }
  get time(): number { return this.now; }

  /** Records a sample if >= 1/60 s of sim time has passed. `fill` writes channel values in place. */
  maybeSample(t: number, fill: (out: Float32Array, offset: number) => void): void {
    if (t < this.now) this.clear(); // sim was reset
    this.now = t;
    while (this.count > 0 && this.timeAt(0) < t - WINDOW_S) this.count--;
    if (t - this.lastT + 1e-10 < 1 / SAMPLE_HZ) return;
    this.lastT = t;
    fill(this.data, this.head * this.channels);
    this.times[this.head] = t;
    this.head = (this.head + 1) % CAP;
    if (this.count < CAP) this.count++;
  }

  /** i = 0 is oldest. */
  at(i: number, ch: number): number {
    const idx = (this.head - this.count + i + CAP * 2) % CAP;
    return this.data[idx * this.channels + ch]!;
  }

  timeAt(i: number): number {
    return this.times[(this.head - this.count + i + CAP * 2) % CAP]!;
  }
}

export interface Series { channel: number; color: string; dashed?: boolean; width?: number }

const NICE = [0.5, 1, 2, 5, 10, 20, 45, 90, 180, 360];
const NICE_M = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5];
const GRID_FRACTIONS = [-1, -0.5, 0, 0.5, 1];
const DASHED = [4, 4];
const SOLID: number[] = [];

export interface PlotOptions { unit: "deg" | "m"; scale: number; symmetric: boolean }

export class PlotCanvas {
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private w = 1;
  private h = 1;
  private range = 5; // smoothed half-range
  series: Series[] = [];

  constructor(private canvas: HTMLCanvasElement, private opts: PlotOptions) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D not supported");
    this.ctx = ctx;
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize(): void {
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, r.width);
    this.h = Math.max(1, r.height);
    const pw = Math.round(this.w * this.dpr), ph = Math.round(this.h * this.dpr);
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw;
      this.canvas.height = ph;
    }
  }

  draw(hist: History): void {
    const ctx = this.ctx;
    const { w, h } = this;
    const padL = 38, padR = 8, padT = 8, padB = 16;
    const pw = w - padL - padR, ph = h - padT - padB;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // autoscale: smallest "nice" half-range that holds the data (grows instantly, shrinks slowly)
    let maxAbs = 0;
    const n = hist.length;
    for (const s of this.series) {
      for (let i = 0; i < n; i++) {
        const v = Math.abs(hist.at(i, s.channel) * this.opts.scale);
        if (v > maxAbs) maxAbs = v;
      }
    }
    const table = this.opts.unit === "deg" ? NICE : NICE_M;
    let target = table[table.length - 1]!;
    for (const c of table) if (c >= maxAbs * 1.05) { target = c; break; }
    this.range = target > this.range ? target : this.range + (target - this.range) * 0.05;
    const R = this.range;

    // grid + labels
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (const f of GRID_FRACTIONS) {
      const y = padT + ph * (0.5 - f / 2);
      ctx.strokeStyle = f === 0 ? "#34404f" : THEME.grid;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke();
      ctx.fillStyle = THEME.dim;
      const val = f * R;
      ctx.fillText(Math.abs(val) < 1 && val !== 0 ? val.toFixed(2) : val.toFixed(Math.abs(val) < 10 && val % 1 !== 0 ? 1 : 0), padL - 4, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    for (let s = 0; s <= WINDOW_S; s += 2) {
      const x = padL + pw * (s / WINDOW_S);
      ctx.fillStyle = THEME.dim;
      ctx.fillText(s === WINDOW_S ? "now" : `-${WINDOW_S - s}s`, Math.min(Math.max(x, padL + 8), padL + pw - 8), h - 3);
    }
    if (n < 2) return;

    ctx.save();
    ctx.beginPath(); ctx.rect(padL, padT, pw, ph); ctx.clip();
    ctx.lineJoin = "round";
    for (const s of this.series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width ?? 1.6;
      ctx.setLineDash(s.dashed ? DASHED : SOLID);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = padL + pw * (1 - (hist.time - hist.timeAt(i)) / WINDOW_S);
        const y = padT + ph * (0.5 - hist.at(i, s.channel) * this.opts.scale / (2 * R));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.restore();
    ctx.setLineDash(SOLID);
  }
}

/** Convenience: series for n link angles. */
export function angleSeries(n: number): Series[] {
  const out: Series[] = [];
  for (let i = 0; i < n; i++) out.push({ channel: i, color: linkColor(i) });
  return out;
}
