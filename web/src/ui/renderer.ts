import type { SimStatus, SimulationApi } from "../sim/types";
import { THEME, linkColor } from "./colors";

const MAX_FLASH = 8;
const FLASH_SECONDS = 0.6;

interface Flash { active: boolean; link: number; dir: number; t0: number }

export interface LinkHit { link: number; dir: 1 | -1 }

/** Canvas 2D renderer for the rig. All geometry is in world metres (x right, z up, rail at z = 0). */
export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private w = 1; // css px
  private h = 1;
  private scale = 100; // px per metre
  private ox = 0;
  private oy = 0;
  private sim: SimulationApi;
  private flashes: Flash[] = [];
  private lastPts: Array<[number, number]> = [];
  private height = 1;
  private half = 0.6;

  constructor(private canvas: HTMLCanvasElement, sim: SimulationApi) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D not supported");
    this.ctx = ctx;
    this.sim = sim;
    for (let i = 0; i < MAX_FLASH; i++) this.flashes.push({ active: false, link: 0, dir: 1, t0: 0 });
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.setSim(sim);
  }

  setSim(sim: SimulationApi): void {
    this.sim = sim;
    const rig = sim.rig;
    this.half = rig.cart.railHalfLength;
    this.height = rig.links.reduce((s, l) => s + l.length, 0);
    this.lastPts = sim.linkPoints();
    for (const f of this.flashes) f.active = false;
    this.layout();
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
    this.layout();
  }

  private layout(): void {
    const margin = 0.16;
    this.oy = this.h - Math.min(70, this.h * 0.14);
    const sx = this.w / (2 * (this.half + margin));
    const sy = (this.oy - 34) / (this.height + 0.05);
    this.scale = Math.max(20, Math.min(sx, sy));
    this.ox = this.w / 2;
  }

  worldToScreenX(x: number): number { return this.ox + x * this.scale; }
  screenToWorldX(px: number): number { return (px - this.ox) / this.scale; }
  /** Pixel position of a canvas-relative client point. */
  canvasPoint(clientX: number, clientY: number): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [clientX - r.left, clientY - r.top];
  }

  /** Nearest link within `radiusPx` of the point; dir = which side of the link the point is on. */
  hitLink(px: number, py: number, radiusPx: number): LinkHit | null {
    const pts = this.lastPts;
    let best = -1, bestD = radiusPx, bestSide: 1 | -1 = 1;
    for (let i = 0; i + 1 < pts.length; i++) {
      const ax = this.ox + pts[i]![0] * this.scale, ay = this.oy - pts[i]![1] * this.scale;
      const bx = this.ox + pts[i + 1]![0] * this.scale, by = this.oy - pts[i + 1]![1] * this.scale;
      const dx = bx - ax, dy = by - ay;
      const len2 = dx * dx + dy * dy || 1;
      const u = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
      const cx = ax + u * dx, cy = ay + u * dy;
      const d = Math.hypot(px - cx, py - cy);
      if (d < bestD) { bestD = d; best = i; bestSide = px >= cx ? 1 : -1; }
    }
    return best < 0 ? null : { link: best, dir: bestSide };
  }

  flashPush(link: number, dir: number): void {
    let f = this.flashes.find((q) => !q.active);
    if (!f) f = this.flashes[0]!;
    f.active = true; f.link = link; f.dir = dir; f.t0 = performance.now();
  }

  draw(st: SimStatus, nowMs: number): void {
    const ctx = this.ctx;
    const { scale, ox, oy } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const half = this.half;
    const sX = (x: number) => ox + x * scale;
    const sZ = (z: number) => oy - z * scale;
    const railY = oy + 8;

    // faint 10 cm ticks across the whole rail
    ctx.strokeStyle = THEME.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = -Math.floor(half * 10); k <= Math.floor(half * 10); k++) {
      const px = sX(k / 10);
      ctx.moveTo(px, 10);
      ctx.lineTo(px, this.h - 10);
    }
    ctx.stroke();

    // rail + end stops + usable-travel marks
    const stop = half + 0.05;
    ctx.strokeStyle = THEME.rail;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(sX(-stop), railY);
    ctx.lineTo(sX(stop), railY);
    ctx.stroke();
    ctx.fillStyle = THEME.stop;
    for (const s of [-stop, stop]) ctx.fillRect(sX(s) - 5, oy - 10, 10, 30);
    const lim = 0.8 * half;
    ctx.strokeStyle = THEME.target;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const s of [-lim, lim]) { ctx.moveTo(sX(s), railY + 8); ctx.lineTo(sX(s), railY + 26); }
    ctx.stroke();
    ctx.fillStyle = THEME.dim;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("usable travel", sX(0), railY + 38);
    ctx.fillText(`${(-lim).toFixed(2)} m`, sX(-lim), railY + 38);
    ctx.fillText(`${lim.toFixed(2)} m`, sX(lim), railY + 38);
    ctx.strokeStyle = THEME.dim;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sX(0), railY + 8); ctx.lineTo(sX(0), railY + 20); ctx.stroke();

    // target marker
    const tx = sX(st.xRef);
    ctx.strokeStyle = THEME.target;
    ctx.fillStyle = THEME.target;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 5]);
    ctx.beginPath(); ctx.moveTo(tx, railY - 6); ctx.lineTo(tx, sZ(this.height)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(tx, railY + 4); ctx.lineTo(tx - 8, railY + 20); ctx.lineTo(tx + 8, railY + 20);
    ctx.closePath(); ctx.fill();
    ctx.textAlign = "center";
    // Below the rail labels so it never collides with the pendulum tip.
    ctx.fillText(`target ${(st.xRef * 100).toFixed(0)} cm`, tx, railY + 56);

    // controller's estimate (Kalman filter) as a faint ghost chain behind the true one
    const links = this.sim.rig.links;
    const n = links.length;
    const th = st.thetaEstimate;
    let ex = st.x, ez = 0;
    ctx.globalAlpha = 0.4;
    for (let i = 0; i < n; i++) {
      const a = th[i] ?? 0;
      const nx = ex + links[i]!.length * Math.sin(a);
      const nz = ez + links[i]!.length * Math.cos(a);
      ctx.strokeStyle = linkColor(i);
      ctx.lineWidth = Math.max(2, 0.008 * scale);
      ctx.beginPath(); ctx.moveTo(sX(ex), sZ(ez)); ctx.lineTo(sX(nx), sZ(nz)); ctx.stroke();
      ex = nx; ez = nz;
    }
    ctx.globalAlpha = 1;

    // cart
    const pts = this.sim.linkPoints();
    this.lastPts = pts;
    const cw = 0.12 * scale, ch = 0.055 * scale;
    const cx = sX(pts[0]![0]);
    ctx.fillStyle = THEME.cart;
    roundRect(ctx, cx - cw / 2, oy - ch / 2 + 2, cw, ch, 5);
    ctx.fill();
    ctx.fillStyle = THEME.wheel;
    for (const d of [-cw / 3, cw / 3]) {
      ctx.beginPath(); ctx.arc(cx + d, railY - 1, Math.max(3, ch / 4), 0, Math.PI * 2); ctx.fill();
    }

    // true links + joints
    const lw = Math.max(4, 0.014 * scale);
    for (let i = 0; i < n; i++) {
      ctx.strokeStyle = linkColor(i);
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(sX(pts[i]![0]), sZ(pts[i]![1]));
      ctx.lineTo(sX(pts[i + 1]![0]), sZ(pts[i + 1]![1]));
      ctx.stroke();
    }
    for (let i = 0; i < pts.length; i++) {
      const px = sX(pts[i]![0]), pz = sZ(pts[i]![1]);
      ctx.fillStyle = THEME.fg;
      ctx.beginPath(); ctx.arc(px, pz, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = THEME.bg;
      ctx.beginPath(); ctx.arc(px, pz, 2, 0, Math.PI * 2); ctx.fill();
    }

    // push arrows
    for (const f of this.flashes) {
      if (!f.active) continue;
      const age = (nowMs - f.t0) / 1000;
      if (age > FLASH_SECONDS) { f.active = false; continue; }
      const p = pts[f.link + 1];
      if (!p) continue;
      const alpha = 1 - age / FLASH_SECONDS;
      const px = sX(p[0]), pz = sZ(p[1]);
      const len = 46 + 14 * (1 - alpha);
      const x0 = px - f.dir * 4, x1 = px + f.dir * len;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = "#ffffff";
      ctx.fillStyle = "#ffffff";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x0 - f.dir * 28, pz); ctx.lineTo(x1 - f.dir * 8, pz); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, pz); ctx.lineTo(x1 - f.dir * 12, pz - 7); ctx.lineTo(x1 - f.dir * 12, pz + 7); ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }

    // scale bar
    const bar = 0.1 * scale, bx = 16, by = this.h - 18;
    ctx.strokeStyle = THEME.fg; ctx.fillStyle = THEME.fg; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bx, by); ctx.lineTo(bx + bar, by);
    ctx.moveTo(bx, by - 4); ctx.lineTo(bx, by + 4);
    ctx.moveTo(bx + bar, by - 4); ctx.lineTo(bx + bar, by + 4);
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.fillText("0.1 m", bx, by - 8);
  }
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
