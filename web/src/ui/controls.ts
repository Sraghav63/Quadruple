import type { Renderer } from "./renderer";
import { $ } from "./dom";

export interface Actions {
  togglePause(): void;
  reset(): void;
  toggleController(): void;
  nudgeTarget(dx: number): void;
  setTargetX(x: number): void;
  push(link: number, dir: 1 | -1): void;
}

const TARGET_STEP = 0.05;
const DRAG_PX = 8;
const TAP_MS = 350;

/** Pointer, keyboard and toolbar-button wiring. */
export function bindControls(canvas: HTMLCanvasElement, renderer: Renderer, act: Actions): void {
  let pointerId: number | null = null;
  let startX = 0, startY = 0, startTime = 0;
  let hit: ReturnType<Renderer["hitLink"]> = null;
  let dragging = false;

  const move = (e: PointerEvent) => {
    if (Math.hypot(e.clientX - startX, e.clientY - startY) > DRAG_PX) dragging = true;
    if (dragging) act.setTargetX(renderer.screenToWorldX(renderer.canvasPoint(e.clientX, e.clientY)[0]));
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (!e.isPrimary || pointerId !== null || (e.button !== 0 && e.pointerType === "mouse")) return;
    pointerId = e.pointerId;
    startX = e.clientX; startY = e.clientY; startTime = e.timeStamp;
    const [px, py] = renderer.canvasPoint(e.clientX, e.clientY);
    hit = renderer.hitLink(px, py, e.pointerType === "mouse" ? 18 : 30);
    dragging = !hit;
    canvas.setPointerCapture(e.pointerId);
    if (dragging) act.setTargetX(renderer.screenToWorldX(px));
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (e.pointerId !== pointerId) return;
    move(e);
  });
  const end = (e: PointerEvent) => {
    if (e.pointerId !== pointerId) return;
    pointerId = null; dragging = false; hit = null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup", (e) => {
    if (e.pointerId !== pointerId) return;
    move(e);
    if (!dragging && hit && e.timeStamp - startTime < TAP_MS) act.push(hit.link, hit.dir);
    end(e);
  });
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("lostpointercapture", end);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
    if (/^Digit[1-9]$/.test(e.code)) {
      act.push(parseInt(e.code.slice(5), 10) - 1, e.shiftKey ? -1 : 1);
      e.preventDefault();
      return;
    }
    switch (e.key) {
      case "ArrowLeft": case "a": case "A": act.nudgeTarget(-TARGET_STEP); break;
      case "ArrowRight": case "d": case "D": act.nudgeTarget(TARGET_STEP); break;
      case " ": act.toggleController(); break;
      case "r": case "R": act.reset(); break;
      case "p": case "P": act.togglePause(); break;
      default: return;
    }
    e.preventDefault();
    if (t && t.tagName === "BUTTON") t.blur(); // keep Space from also "clicking" a focused button
  });

  $("btn-pause").addEventListener("click", () => act.togglePause());
  $("btn-reset").addEventListener("click", () => act.reset());
  $("fallen-reset").addEventListener("click", () => act.reset());
  $("btn-ctrl").addEventListener("click", () => act.toggleController());
  $("btn-left").addEventListener("click", () => act.nudgeTarget(-TARGET_STEP));
  $("btn-right").addEventListener("click", () => act.nudgeTarget(TARGET_STEP));
}
