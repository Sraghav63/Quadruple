import "./styles.css";
import { createSimulation, loadPreset, loadPresetIndex, DEFAULT_KNOBS } from "./sim";
import type { SimulationApi } from "./sim";
import { $, el } from "./ui/dom";
import { Renderer } from "./ui/renderer";
import { History, PlotCanvas, angleSeries } from "./ui/plots";
import { Hud } from "./ui/hud";
import { Panel } from "./ui/panel";
import { bindControls } from "./ui/controls";
import { THEME } from "./ui/colors";
import { parseHash, resolveConfiguration } from "./ui/location";

const REPO_URL = ""; // set to the GitHub repo URL to show the link in the header
const RAD2DEG = 180 / Math.PI;
const PUSH_SECONDS = 0.02;

let sim: SimulationApi;
let renderer: Renderer;
let hud: Hud;
let panel: Panel;
let hist = new History(1);
let paused = false;
let speed = 1;
let tapForce = 0.08;
let loadToken = 0;

const canvas = $<HTMLCanvasElement>("scene");
const fallenOverlay = $("fallen");
const loadError = $("load-error");
const btnPause = $("btn-pause");
const btnCtrl = $("btn-ctrl");
const anglesPlot = new PlotCanvas($<HTMLCanvasElement>("plot-angles"), { unit: "deg", scale: RAD2DEG, symmetric: true });
const xPlot = new PlotCanvas($<HTMLCanvasElement>("plot-x"), { unit: "m", scale: 1, symmetric: true });

if (REPO_URL) {
  const a = $<HTMLAnchorElement>("repo-link");
  a.href = REPO_URL;
  a.hidden = false;
}

function syncButtons(): void {
  btnPause.textContent = paused ? "resume" : "pause";
  const on = sim.controllerEnabled;
  btnCtrl.textContent = on ? "controller on" : "controller off";
  btnCtrl.classList.toggle("on", on);
}

function resetSim(): void {
  // A reset means "try again": re-arm the controller so it doesn't fall straight back over.
  sim.controllerEnabled = true;
  sim.reset();
  hist.clear();
  panel.syncConfiguration();
  replaceHash(sim.preset.id);
  syncButtons();
}

function push(link: number, dir: 1 | -1): void {
  if (link >= sim.rig.links.length) return;
  sim.push(link, dir * tapForce, PUSH_SECONDS);
  renderer.flashPush(link, dir);
}

function describe(preset: SimulationApi["preset"]): void {
  const body = $("about-body");
  body.replaceChildren();
  const p = (html: string) => { const e = el("p"); e.innerHTML = html; body.append(e); };
  p("The cart can only slide left and right along the rail; everything else has to be balanced by moving it. " +
    "Drag on the scene to move the target, or tap a link to give it a small push.");
  p("A linear-quadratic regulator (LQR) with a Kalman-style observer runs at 1 kHz with 1 ms of compute delay and " +
    "14-bit joint encoders, like the real build. The faint ghost chain is what the controller thinks the pendulum is doing.");
  p("Joint stiction (friction that grips at rest) makes the pendulum wander a few degrees instead of sitting perfectly still. " +
    "A quadruple pendulum only tolerates light taps: about 0.1 N for 20 ms on the top link.");
  const cfg = preset.controller.configuration;
  const poles = preset.configurations[cfg].unstablePoles;
  const times = poles.map((q) => `${(Math.LN2 / q * 1000).toFixed(0)} ms`);
  if (poles.length) p(`This rig (${preset.name}) in the all-up configuration ${cfg} has ${poles.length} unstable modes. ` +
    `Each one doubles in size every ${times.join(", ")} (ln 2 / pole, fastest first), which is why it needs a fast, accurate controller.`);
  else p(`Configuration ${cfg} has zero unstable modes: the all-down chain is naturally stable, and the controller just damps its motion.`);
  p("The simulation starts with every link hanging down at rest. Configuration letters run from the bottom link to the top: U is up, D is down. " +
    "Choose a configuration to move there live. The cart first centres and settles, then follows offline-planned motions tracked by a time-varying LQR + Kalman filter. " +
    "A route may use several motions through intermediate configurations. Reset returns the chain to hanging down.");
  p("Keys: <kbd>←</kbd>/<kbd>→</kbd> or <kbd>A</kbd>/<kbd>D</kbd> move the target, <kbd>1</kbd>-<kbd>4</kbd> push a link right " +
    "(<kbd>Shift</kbd> for left), <kbd>Space</kbd> toggles the controller, <kbd>R</kbd> resets, <kbd>P</kbd> pauses. " +
    "Clicking a link pushes it toward the side you clicked.");
  if (REPO_URL) {
    const e = el("p");
    const a = el("a", "", "Source on GitHub");
    a.href = REPO_URL;
    a.target = "_blank";
    a.rel = "noopener";
    e.append(a);
    body.append(e);
  }
}

function install(newSim: SimulationApi): void {
  if (sim) sim.controllerEnabled = false; // cancel any pending design for the old preset
  sim = newSim;
  const n = sim.rig.links.length;
  hist = new History(n + 2);
  anglesPlot.series = angleSeries(n);
  xPlot.series = [
    { channel: n, color: THEME.cart, width: 2 },
    { channel: n + 1, color: THEME.target, dashed: true, width: 1.4 },
  ];
  if (renderer) renderer.setSim(sim); else renderer = new Renderer(canvas, sim);
  if (hud) hud.setSim(sim); else hud = new Hud(sim);
  panel.setSim(sim);
  describe(sim.preset);
  syncButtons();
}

async function selectConfiguration(configuration: string): Promise<void> {
  const current = sim;
  const request = current.requestConfiguration(configuration);
  panel.updateTransition(current.status());
  await request;
  if (current !== sim) return;
  panel.syncConfiguration();
  replaceHash(sim.preset.id);
}

async function selectPreset(id: string, configuration?: string): Promise<void> {
  const token = ++loadToken;
  try {
    if (sim && sim.preset.id === id) {
      await selectConfiguration(resolveConfiguration(sim.preset, configuration));
      loadError.hidden = true;
      return;
    }
    const preset = await loadPreset(id);
    if (token !== loadToken) return;
    loadError.hidden = true;
    const newSim = createSimulation(preset);
    install(newSim);
    panel.setActive(id);
    void selectConfiguration(resolveConfiguration(preset, configuration));
  } catch (err) {
    if (token !== loadToken) return;
    loadError.hidden = false;
    loadError.textContent = `Could not load preset "${id}": ${err instanceof Error ? err.message : String(err)}`;
    if (sim) panel.setActive(sim.preset.id);
  }
}

function replaceHash(id: string): void {
  window.history.replaceState(null, "", `#${id}/${sim.status().target}`);
}

panel = new Panel({
  onPreset: (id) => { void selectPreset(id); },
  onConfiguration: (cfg) => { void selectConfiguration(cfg); },
  onSpeed: (v) => { speed = v; },
  onTapForce: (f) => { tapForce = f; },
  onPush: push,
  onKnobsReset: () => {
    sim.setKnobs(DEFAULT_KNOBS(sim.preset));
    panel.syncKnobs();
  },
});

// ---------- frame loop ----------
let lastFrame = performance.now();
let perfWall = 0, perfSim = 0, perfSteps = 0;
let perfText = "";
let hudTick = 0;

function frame(now: number): void {
  const wall = Math.max(0, (now - lastFrame) / 1000);
  lastFrame = now;
  if (sim) {
    const t0 = sim.status().t;
    let steps = 0;
    if (!paused) steps = sim.advance(Math.min(wall, 0.05) * speed);
    const st = sim.status();

    perfWall += wall; perfSim += st.t - t0 > 0 ? st.t - t0 : 0; perfSteps += steps;
    if (perfWall >= 0.5) {
      perfText = paused ? "paused" : `${(perfSim / perfWall).toFixed(2)}× real time · ${Math.round(perfSteps / perfWall)} steps/s`;
      perfWall = perfSim = perfSteps = 0;
    }

    const n = sim.rig.links.length;
    hist.maybeSample(st.t, (out, o) => {
      for (let i = 0; i < n; i++) out[o + i] = st.thetaDeviation[i] ?? 0;
      out[o + n] = st.x - st.xRef;
      out[o + n + 1] = 0;
    });

    renderer.draw(st, now);
    anglesPlot.draw(hist);
    xPlot.draw(hist);
    if (fallenOverlay.hidden !== !st.fallen) fallenOverlay.hidden = !st.fallen;
    if (hudTick++ % 3 === 0) { hud.update(st, perfText); panel.updateTransition(st); }
  }
  requestAnimationFrame(frame);
}

async function start(): Promise<void> {
  const list = await loadPresetIndex();
  const { presetId: id, configuration } = parseHash(location.hash, list);
  panel.setPresets(list, id);
  const preset = await loadPreset(id);
  const newSim = createSimulation(preset);
  install(newSim);
  void selectConfiguration(resolveConfiguration(preset, configuration));

  bindControls(canvas, renderer, {
    togglePause: () => { paused = !paused; syncButtons(); },
    reset: resetSim,
    toggleController: () => { sim.controllerEnabled = !sim.controllerEnabled; syncButtons(); },
    nudgeTarget: (dx) => sim.setReference(sim.status().xRef + dx),
    setTargetX: (x) => sim.setReference(x),
    push,
  });
  window.addEventListener("hashchange", () => {
    const h = parseHash(location.hash, list);
    if (h.presetId !== sim.preset.id || h.configuration !== sim.status().target) void selectPreset(h.presetId, h.configuration);
  });
  new ResizeObserver(() => { renderer.resize(); anglesPlot.resize(); xPlot.resize(); }).observe($("stage-wrap"));
  renderer.resize(); anglesPlot.resize(); xPlot.resize();
  lastFrame = performance.now();
  requestAnimationFrame(frame);
}

start().catch((err) => {
  loadError.hidden = false;
  loadError.textContent = `Failed to start: ${err instanceof Error ? err.message : String(err)}`;
});
