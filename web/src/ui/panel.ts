import type { PresetSummary, RealityKnobs, SimulationApi } from "../sim/types";
import { $, el } from "./dom";
import { linkColor } from "./colors";

interface KnobSpec {
  key: keyof RealityKnobs;
  label: string;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
  tip: string;
}

const KNOBS: KnobSpec[] = [
  { key: "frictionScale", label: "Joint friction", min: 0, max: 4, step: 0.1, fmt: (v) => `×${v.toFixed(1)}`,
    tip: "How sticky the hinges are. Real bearings have stiction, which makes the pendulum wander a few degrees. 0 is a perfect frictionless joint." },
  { key: "massScale", label: "Link mass", min: 0.7, max: 1.3, step: 0.01, fmt: (v) => `×${v.toFixed(2)}`,
    tip: "Makes the real rods heavier or lighter than the controller believes. The controller keeps its original numbers, so this is pure model error." },
  { key: "sensorNoiseScale", label: "Sensor noise", min: 0, max: 5, step: 0.1, fmt: (v) => `×${v.toFixed(1)}`,
    tip: "Jitter on the angle encoders. More noise makes the estimate (the faint ghost chain) shakier and the cart twitchier." },
  { key: "encoderBits", label: "Encoder resolution", min: 10, max: 21, step: 1, fmt: (v) => `${v.toFixed(0)} bit`,
    tip: "How finely each joint angle is measured. Fewer bits means coarser steps; at 10 bits the top links are nearly invisible to the controller." },
  { key: "extraDelaySteps", label: "Extra latency", min: 0, max: 10, step: 1, fmt: (v) => `${v.toFixed(0)} ms`,
    tip: "Added delay between measuring and acting, on top of the 1 ms the controller already knows about. Delay is the classic way to lose balance." },
  { key: "driveBandwidthScale", label: "Drive bandwidth", min: 0.3, max: 2, step: 0.05, fmt: (v) => `×${v.toFixed(2)}`,
    tip: "How quickly the servo drive follows the acceleration command. Lower means a sluggish motor that lags behind what the controller asks for." },
  { key: "accelLimit", label: "Accel limit", min: 5, max: 60, step: 1, fmt: (v) => `${v.toFixed(0)} m/s²`,
    tip: "The most the cart is allowed to accelerate. A weaker drive can't make the large, fast moves a quadruple pendulum sometimes needs." },
];

export interface PanelCallbacks {
  onPreset(id: string): void;
  onSpeed(v: number): void;
  onTapForce(n: number): void;
  onPush(link: number, dir: 1 | -1): void;
  onKnobsReset(): void;
}

export class Panel {
  private select = $<HTMLSelectElement>("preset");
  private desc = $("preset-desc");
  private knobsBox = $("knobs");
  private pushRow = $("push-row");
  private inputs = new Map<keyof RealityKnobs, { input: HTMLInputElement; out: HTMLElement; spec: KnobSpec }>();
  private sim: SimulationApi | null = null;
  private summaries: PresetSummary[] = [];
  tapForce = 0.08;
  speed = 1;

  constructor(private cb: PanelCallbacks) {
    this.select.addEventListener("change", () => cb.onPreset(this.select.value));

    const speed = $<HTMLInputElement>("speed");
    const speedVal = $("speed-val");
    speed.addEventListener("input", () => {
      this.speed = parseFloat(speed.value);
      speedVal.textContent = `${this.speed.toFixed(2)}×`;
      cb.onSpeed(this.speed);
    });
    const tap = $<HTMLInputElement>("tap");
    const tapVal = $("tap-val");
    tap.addEventListener("input", () => {
      this.tapForce = parseFloat(tap.value);
      tapVal.textContent = `${this.tapForce.toFixed(2)} N`;
      cb.onTapForce(this.tapForce);
    });
    for (const i of [speed, tap]) i.addEventListener("change", () => i.blur());
    $("knobs-reset").addEventListener("click", () => cb.onKnobsReset());

    for (const spec of KNOBS) {
      const wrap = el("div", "knob");
      const row = el("div", "row");
      const tip = el("button", "tip", "?");
      tip.type = "button";
      tip.dataset.tip = spec.tip;
      tip.setAttribute("aria-label", `${spec.label}: ${spec.tip}`);
      const out = el("b");
      row.append(el("span", "", spec.label), tip, out);
      const input = el("input");
      input.type = "range";
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
      input.setAttribute("aria-label", spec.label);
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        out.textContent = spec.fmt(v);
        this.sim?.setKnobs({ [spec.key]: v } as Partial<RealityKnobs>);
      });
      input.addEventListener("change", () => input.blur());
      wrap.append(row, input);
      this.knobsBox.append(wrap);
      this.inputs.set(spec.key, { input, out, spec });
    }
  }

  setPresets(list: PresetSummary[], activeId: string): void {
    this.summaries = list;
    this.select.replaceChildren(...list.map((p) => {
      const o = el("option", "", p.name);
      o.value = p.id;
      return o;
    }));
    this.select.value = activeId;
    this.showDesc(activeId);
  }

  private showDesc(id: string): void {
    this.desc.textContent = this.summaries.find((p) => p.id === id)?.description ?? "";
  }

  setActive(id: string): void { this.select.value = id; this.showDesc(id); }

  /** Rebind to a (new) simulation: slider ranges and positions come from its knobs. */
  setSim(sim: SimulationApi): void {
    this.sim = sim;
    this.syncKnobs();
    this.pushRow.replaceChildren();
    sim.rig.links.forEach((_, i) => {
      const row = el("div", "push-item");
      const name = el("span", "name");
      const sw = el("span", "swatch");
      sw.style.background = linkColor(i);
      name.append(sw, document.createTextNode(`Push link ${i + 1}`));
      const l = el("button", "btn", "←");
      const r = el("button", "btn", "→");
      l.type = r.type = "button";
      l.title = `Push link ${i + 1} left (Shift+${i + 1})`;
      r.title = `Push link ${i + 1} right (${i + 1})`;
      l.setAttribute("aria-label", l.title);
      r.setAttribute("aria-label", r.title);
      l.addEventListener("click", () => this.cb.onPush(i, -1));
      r.addEventListener("click", () => this.cb.onPush(i, 1));
      row.append(name, l, r);
      this.pushRow.append(row);
    });
  }

  /** Copy sim.knobs into the sliders. */
  syncKnobs(): void {
    if (!this.sim) return;
    const k = this.sim.knobs;
    for (const { input, out, spec } of this.inputs.values()) {
      const v = k[spec.key];
      input.min = String(spec.min);
      input.max = String(spec.max);
      // widen the slider if the preset's own default is outside the nominal range
      if (v < spec.min) input.min = String(v);
      if (v > spec.max) input.max = String(v);
      input.value = String(v);
      out.textContent = spec.fmt(v);
    }
  }
}
