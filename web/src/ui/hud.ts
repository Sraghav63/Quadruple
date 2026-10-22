import type { SimStatus, SimulationApi } from "../sim/types";
import { $, el, setText } from "./dom";
import { linkColor } from "./colors";

const RAD2DEG = 180 / Math.PI;

export class Hud {
  private t = $("h-t");
  private x = $("h-x");
  private v = $("h-v");
  private a = $("h-a");
  private f = $("h-f");
  private c = $("h-c");
  private barF = $("bar-f");
  private barC = $("bar-c");
  private status = $("status");
  private perf = $("perf");
  private angleBox = $("angles");
  private configuration = $("h-configuration");
  private angleEls: HTMLElement[] = [];
  private sim: SimulationApi;

  constructor(sim: SimulationApi) {
    this.sim = sim;
    this.setSim(sim);
  }

  setSim(sim: SimulationApi): void {
    this.sim = sim;
    this.angleBox.replaceChildren();
    this.angleEls = [];
    sim.rig.links.forEach((_, i) => {
      const row = el("span", "key");
      const dot = el("i", "dot");
      dot.style.background = linkColor(i);
      row.title = `Link ${i + 1}: deviation from target`;
      const v = el("span", "v", "0.0°");
      row.append(dot, v);
      this.angleBox.append(row);
      this.angleEls.push(v);
    });
  }

  update(st: SimStatus, perfText: string): void {
    const drive = this.sim.rig.drive;
    const accLim = this.sim.knobs.accelLimit;
    setText(this.t, `${st.t.toFixed(2)} s`);
    setText(this.configuration, st.configuration);
    setText(this.x, `${st.x.toFixed(3)} m`);
    setText(this.v, `${st.v.toFixed(2)} m/s`);
    setText(this.a, `${st.a.toFixed(1)} m/s²`);
    setText(this.f, `${st.force.toFixed(0)} / ${drive.forceLimit.toFixed(0)} N`);
    setText(this.c, `${st.command.toFixed(1)} / ${accLim.toFixed(0)} m/s²`);
    setBar(this.barF, st.force / drive.forceLimit);
    setBar(this.barC, st.command / accLim);
    for (let i = 0; i < this.angleEls.length; i++) {
      setText(this.angleEls[i]!, `${((st.thetaDeviation[i] ?? 0) * RAD2DEG).toFixed(1)}°`);
    }
    let text = "balancing", cls = "pill ok";
    if (st.mode === 'center') { text = st.designProgress !== null ? 'preparing motion' : 'centring'; cls = 'pill warn'; }
    if (st.mode === 'transition') text = `moving · ${Math.round(100 * (st.transitionProgress ?? 0))}%`;
    if (st.fallen) { text = "fallen"; cls = "pill bad"; }
    else if (st.hitRailEnd) { text = "hit rail end"; cls = "pill warn"; }
    else if (!st.controllerEnabled) { text = "controller off"; cls = "pill warn"; }
    setText(this.status, text);
    if (this.status.className !== cls) this.status.className = cls;
    setText(this.perf, perfText);
  }
}

function setBar(bar: HTMLElement, frac: number): void {
  const f = Math.max(-1, Math.min(1, frac));
  const i = bar.firstElementChild as HTMLElement;
  i.style.width = `${Math.abs(f) * 50}%`;
  i.style.transform = f < 0 ? "translateX(-100%)" : "none";
  const cls = Math.abs(f) >= 0.98 ? "bar max" : Math.abs(f) > 0.8 ? "bar hot" : "bar";
  if (bar.className !== cls) bar.className = cls;
}
