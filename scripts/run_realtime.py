#!/usr/bin/env python
"""Real-time interactive viewer for the quadruple inverted pendulum (LQR)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENSAVER", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from quad_pendulum.params import RigParams
from quad_pendulum.sim import Simulation

PUSH_FORCE = 0.1  # N, horizontal tap at the top of a link (0.2 N already topples links 3-4)
PUSH_DURATION = 0.02  # s
REF_STEP = 0.05  # m per arrow-key press
START_TILT_DEG = 0.05
MAX_CATCHUP = 0.05  # s of wall clock simulated per frame at most
FPS = 60
PLOT_SECONDS = 5.0
PLOT_EVERY = 0.02  # s between plot samples

BG = (24, 27, 32)
PANEL = (33, 37, 44)
FG = (225, 228, 232)
DIM = (140, 146, 155)
GRID = (55, 60, 68)
RAIL = (120, 126, 134)
STOP = (200, 70, 60)
CART = (50, 110, 190)
TARGET = (240, 200, 60)
OK = (80, 200, 120)
BAD = (230, 70, 70)
LINK_COLORS = [(235, 95, 75), (70, 190, 120), (240, 190, 60), (150, 110, 220)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speed", type=float, default=1.0, help="simulation speed factor, 0.1-2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tilt-deg", type=float, default=START_TILT_DEG, help="initial random tilt")
    p.add_argument("--no-controller", action="store_true")
    p.add_argument("--headless", type=float, metavar="SECONDS", help="no window; print a summary")
    p.add_argument("--max-seconds", type=float, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.speed = float(np.clip(args.speed, 0.1, 2.0))
    return args


def make_sim(args: argparse.Namespace) -> Simulation:
    sim = Simulation(RigParams(), seed=args.seed)
    sim.reset(random_tilt_deg=args.tilt_deg)
    sim.controller_enabled = not args.no_controller
    return sim


def run_headless(args: argparse.Namespace) -> int:
    sim = make_sim(args)
    rate = round(sim.rig.timing.control_rate)
    max_theta = max_force = 0.0
    for second in range(1, int(np.ceil(args.headless)) + 1):
        for _ in range(min(rate, round((args.headless - (second - 1)) * rate))):
            sim.step()
            st = sim.status()
            max_theta = max(max_theta, float(np.max(np.abs(np.rad2deg(st["theta"])))))
            max_force = max(max_force, abs(float(sim.plant.state()["force"])))
        print(
            f"t={sim.t:6.2f} s  x={st['x']:+.3f} m  max|theta|={max_theta:6.3f} deg  "
            f"max|force|={max_force:6.2f} N",
            flush=True,
        )
        max_theta = max_force = 0.0
        if sim.fallen:
            break
    if sim.fallen:
        print("FALLEN")
        return 1
    return 0


class Viewer:
    def __init__(self, args: argparse.Namespace) -> None:
        import pygame

        self.pg = pygame
        self.args = args
        self.sim = make_sim(args)
        self.rig = self.sim.rig
        self.dt = self.rig.control_dt
        pygame.init()
        pygame.display.set_caption("Quadruple Inverted Pendulum – LQR")
        self.screen = pygame.display.set_mode((1200, 760), pygame.RESIZABLE)
        pygame.key.set_repeat(250, 60)
        self.font = self.sysfont(15)
        self.small = self.sysfont(12)
        self.big = self.sysfont(30, bold=True)
        self.paused = False
        self.dragging = False
        self.acc = 0.0
        self.rtf = 0.0
        self._rtf_sim0 = 0.0
        self._rtf_wall0 = time.perf_counter()
        self.hist: deque = deque()
        self._next_sample = 0.0
        self.scene_rect = pygame.Rect(0, 0, 1, 1)
        self.scale = 1.0
        self.origin = (0, 0)

    def sysfont(self, size: int, bold: bool = False):
        pg = self.pg
        try:
            return pg.font.SysFont("menlo,dejavusansmono,consolas,couriernew,monospace", size, bold=bold)
        except (RuntimeError, OSError):
            return pg.font.Font(None, size + 6)

    # ---- simulation ------------------------------------------------------
    def reset(self) -> None:
        self.sim.reset(random_tilt_deg=START_TILT_DEG)
        self.sim.set_reference(0.0)
        self.hist.clear()
        self._next_sample = 0.0
        self.acc = 0.0

    def advance(self, wall_dt: float) -> None:
        if self.paused or self.sim.fallen:
            return
        self.acc += min(wall_dt, MAX_CATCHUP) * self.args.speed
        n = int(self.acc / self.dt)
        self.acc -= n * self.dt
        for _ in range(n):
            self.sim.step()
            if self.sim.t >= self._next_sample:
                self._next_sample = self.sim.t + PLOT_EVERY
                st = self.sim.status()
                self.hist.append((st["t"], st["x"], st["x_ref"], np.rad2deg(st["theta"])))
            if self.sim.fallen:
                break
        while self.hist and self.hist[0][0] < self.sim.t - PLOT_SECONDS:
            self.hist.popleft()
        now = time.perf_counter()
        if now - self._rtf_wall0 >= 0.5:
            self.rtf = (self.sim.t - self._rtf_sim0) / (now - self._rtf_wall0)
            self._rtf_sim0, self._rtf_wall0 = self.sim.t, now

    # ---- input -----------------------------------------------------------
    def handle(self, ev) -> bool:
        pg = self.pg
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.KEYDOWN:
            k = ev.key
            if k == pg.K_ESCAPE:
                return False
            if k in (pg.K_LEFT, pg.K_a):
                self.sim.set_reference(self.sim.x_ref - REF_STEP)
            elif k in (pg.K_RIGHT, pg.K_d):
                self.sim.set_reference(self.sim.x_ref + REF_STEP)
            elif k == pg.K_SPACE:
                self.sim.controller_enabled = not self.sim.controller_enabled
            elif k == pg.K_r:
                self.reset()
            elif k == pg.K_p:
                self.paused = not self.paused
            elif pg.K_1 <= k <= pg.K_4:
                sign = -1.0 if ev.mod & pg.KMOD_SHIFT else 1.0
                self.sim.push(k - pg.K_1, sign * PUSH_FORCE, PUSH_DURATION)
        elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
            if self.scene_rect.collidepoint(ev.pos):
                self.dragging = True
                self.set_target_from_mouse(ev.pos[0])
        elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pg.MOUSEMOTION and self.dragging:
            self.set_target_from_mouse(ev.pos[0])
        return True

    def set_target_from_mouse(self, px: int) -> None:
        self.sim.set_reference((px - self.origin[0]) / self.scale)

    # ---- drawing ---------------------------------------------------------
    def w2s(self, x: float, z: float) -> tuple[int, int]:
        return int(self.origin[0] + x * self.scale), int(self.origin[1] - z * self.scale)

    def text(self, s, pos, color=FG, font=None, right=False):
        surf = (font or self.font).render(s, True, color)
        rect = surf.get_rect(topright=pos) if right else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def draw(self) -> None:
        pg = self.pg
        W, H = self.screen.get_size()
        hud_w = 300
        plot_h = min(230, H // 3)
        self.scene_rect = pg.Rect(0, 0, W - hud_w, H - plot_h)
        self.screen.fill(BG)
        self.draw_scene()
        self.draw_hud(pg.Rect(W - hud_w, 0, hud_w, H - plot_h))
        self.draw_plots(pg.Rect(0, H - plot_h, W, plot_h))

    def draw_scene(self) -> None:
        pg, scr, sim, rig = self.pg, self.screen, self.sim, self.rig
        r = self.scene_rect
        half = rig.cart.rail_half_length
        height = sum(link.length for link in rig.links)
        margin = 0.18
        self.scale = min(r.width / (2 * (half + margin)), r.height / (height + 0.45))
        self.origin = (r.centerx, r.bottom - int(0.2 * self.scale) - 10)
        pts = sim.plant.link_points()
        st = sim.status()
        ox, oy = self.origin

        # rail with end stops and usable-travel markers
        a, b = self.w2s(-half - 0.05, 0), self.w2s(half + 0.05, 0)
        pg.draw.line(scr, RAIL, (a[0], oy + 6), (b[0], oy + 6), 6)
        for sx in (-half - 0.05, half + 0.05):
            px = self.w2s(sx, 0)[0]
            pg.draw.rect(scr, STOP, (px - 5, oy - 8, 10, 28))
        lim = 0.8 * half
        for sx in (-lim, lim):
            px = self.w2s(sx, 0)[0]
            pg.draw.line(scr, TARGET, (px, oy + 14), (px, oy + 30), 2)
        self.text("usable", (self.w2s(-lim, 0)[0] - 18, oy + 32), DIM, self.small)
        pg.draw.line(scr, DIM, (ox, oy + 14), (ox, oy + 24), 1)

        # cart target marker
        tx = self.w2s(st["x_ref"], 0)[0]
        pg.draw.polygon(scr, TARGET, [(tx, oy + 12), (tx - 8, oy + 28), (tx + 8, oy + 28)])
        pg.draw.line(scr, TARGET, (tx, oy + 12), (tx, oy - int(height * self.scale)), 1)

        # cart
        cx, _cy = self.w2s(pts[0][0], 0)
        cw, ch = int(0.10 * self.scale), int(0.05 * self.scale)
        pg.draw.rect(scr, CART, (cx - cw // 2, oy - ch // 2 + 2, cw, ch), border_radius=4)
        for dx in (-cw // 3, cw // 3):
            pg.draw.circle(scr, (20, 20, 24), (cx + dx, oy + ch // 2 + 4), max(3, ch // 5))

        # links and joints
        for i in range(rig.n_links):
            p0, p1 = self.w2s(*pts[i]), self.w2s(*pts[i + 1])
            pg.draw.line(scr, LINK_COLORS[i % 4], p0, p1, max(4, int(0.014 * self.scale)))
        for p in pts:
            pg.draw.circle(scr, FG, self.w2s(*p), 5)
            pg.draw.circle(scr, BG, self.w2s(*p), 2)

        # scale bar
        bar = int(0.1 * self.scale)
        bx, by = r.left + 20, r.bottom - 22
        pg.draw.line(scr, FG, (bx, by), (bx + bar, by), 2)
        pg.draw.line(scr, FG, (bx, by - 4), (bx, by + 4), 2)
        pg.draw.line(scr, FG, (bx + bar, by - 4), (bx + bar, by + 4), 2)
        self.text("0.1 m", (bx, by - 20), FG, self.small)

        if sim.fallen:
            surf = self.big.render("FALLEN – press R", True, BAD)
            scr.blit(surf, surf.get_rect(center=(r.centerx, r.top + 40)))
        elif self.paused:
            surf = self.big.render("PAUSED", True, DIM)
            scr.blit(surf, surf.get_rect(center=(r.centerx, r.top + 40)))

    def bar(self, x, y, w, value, limit, label):
        pg = self.pg
        self.text(label, (x, y), DIM, self.small)
        by = y + 15
        pg.draw.rect(self.screen, GRID, (x, by, w, 10))
        frac = float(np.clip(value / limit, -1, 1)) if limit else 0.0
        mid = x + w // 2
        col = BAD if abs(frac) > 0.98 else OK
        pg.draw.rect(self.screen, col, (min(mid, mid + frac * w / 2), by, abs(frac) * w / 2, 10))
        pg.draw.line(self.screen, FG, (mid, by - 2), (mid, by + 12), 1)

    def draw_hud(self, r) -> None:
        pg, sim, rig = self.pg, self.sim, self.rig
        pg.draw.rect(self.screen, PANEL, r)
        st = sim.status()
        x, y = r.left + 14, r.top + 12
        w = r.width - 28
        lh = 20
        on = st["controller_enabled"]
        if st["fallen"]:
            status, scol = "FALLEN", BAD
        elif st["hit_rail_end"]:
            status, scol = "RAIL END", (240, 150, 50)
        else:
            status, scol = "BALANCING", OK
        self.text(f"status  {status}", (x, y), scol)
        y += lh
        self.text(f"control {'ON' if on else 'OFF'}", (x, y), OK if on else BAD)
        y += lh + 4
        rtf_col = OK if self.rtf >= 0.95 * self.args.speed else (240, 150, 50)
        self.text(f"t      {st['t']:8.3f} s", (x, y))
        y += lh
        self.text(f"speed  {self.args.speed:.2f}x  RTF {self.rtf:4.2f}", (x, y), rtf_col)
        y += lh + 4
        self.text(f"x      {st['x']:+8.3f} m", (x, y))
        y += lh
        self.text(f"x_ref  {st['x_ref']:+8.3f} m", (x, y), TARGET)
        y += lh
        self.text(f"v      {st['v']:+8.3f} m/s", (x, y))
        y += lh
        self.text(f"a      {st['a']:+8.2f} m/s2", (x, y))
        y += lh + 4
        force = float(sim.plant.state()["force"])
        self.bar(x, y, w, force, rig.drive.force_limit, f"drive force {force:+7.1f} N (lim {rig.drive.force_limit:.0f})")
        y += 36
        self.bar(x, y, w, st["command"], rig.drive.accel_limit, f"accel cmd   {st['command']:+7.2f} (lim {rig.drive.accel_limit:.0f})")
        y += 40
        self.text("link angles (abs)", (x, y), DIM, self.small)
        y += 16
        for i, th in enumerate(np.rad2deg(st["theta"])):
            pg.draw.rect(self.screen, LINK_COLORS[i % 4], (x, y + 4, 10, 10))
            self.text(f"L{i + 1}  {th:+8.3f} deg", (x + 18, y))
            y += lh
        y += 8
        help_lines = [
            "<-/-> A/D  move target 5 cm",
            "mouse drag  set target",
            "1-4 push link right, +Shift left",
            "Space ctrl on/off   R reset",
            "P pause   Esc quit",
        ]
        for line in help_lines:
            if y + 14 < r.bottom:
                self.text(line, (x, y), DIM, self.small)
            y += 15

    def draw_plots(self, r) -> None:
        pg = self.pg
        half_w = r.width // 2
        self.plot(pg.Rect(r.left + 8, r.top + 6, half_w - 12, r.height - 12), "link angles (deg)", "angles")
        self.plot(pg.Rect(r.left + half_w + 4, r.top + 6, half_w - 12, r.height - 12), "cart x (m), target", "cart")

    def plot(self, r, title, kind) -> None:
        pg, scr = self.pg, self.screen
        pg.draw.rect(scr, PANEL, r)
        self.text(title, (r.left + 6, r.top + 3), DIM, self.small)
        inner = pg.Rect(r.left + 6, r.top + 20, r.width - 12, r.height - 26)
        pg.draw.rect(scr, GRID, inner, 1)
        if len(self.hist) < 2:
            return
        t_end = self.sim.t
        t0 = t_end - PLOT_SECONDS
        if kind == "angles":
            series = [(np.array([h[3][i] for h in self.hist]), LINK_COLORS[i % 4]) for i in range(self.rig.n_links)]
            amp = max(1.0, max(float(np.max(np.abs(s))) for s, _ in series) * 1.1)
            lo, hi = -amp, amp
        else:
            xs = np.array([h[1] for h in self.hist])
            rs = np.array([h[2] for h in self.hist])
            series = [(rs, TARGET), (xs, CART)]
            lo, hi = -0.8 * self.rig.cart.rail_half_length - 0.05, 0.8 * self.rig.cart.rail_half_length + 0.05
            lo, hi = min(lo, xs.min() - 0.02), max(hi, xs.max() + 0.02)
        ts = np.array([h[0] for h in self.hist])
        px = inner.left + (ts - t0) / PLOT_SECONDS * inner.width
        zero = inner.bottom - (0 - lo) / (hi - lo) * inner.height
        if inner.top <= zero <= inner.bottom:
            pg.draw.line(scr, GRID, (inner.left, zero), (inner.right, zero), 1)
        self.text(f"{hi:.2g}", (inner.right - 4, inner.top + 1), DIM, self.small, right=True)
        self.text(f"{lo:.2g}", (inner.right - 4, inner.bottom - 14), DIM, self.small, right=True)
        for s, col in series:
            py = inner.bottom - (s - lo) / (hi - lo) * inner.height
            pg.draw.lines(scr, col, False, list(zip(px.tolist(), np.clip(py, inner.top, inner.bottom).tolist())), 2)

    # ---- main loop -------------------------------------------------------
    def run(self) -> int:
        pg = self.pg
        clock = pg.time.Clock()
        start = last = time.perf_counter()
        running = True
        while running:
            for ev in pg.event.get():
                running = self.handle(ev) and running
            now = time.perf_counter()
            self.advance(now - last)
            last = now
            self.draw()
            pg.display.flip()
            clock.tick(FPS)
            if self.args.max_seconds and now - start > self.args.max_seconds:
                break
        pg.quit()
        return 0


def main() -> int:
    args = parse_args()
    if args.headless is not None:
        return run_headless(args)
    if sys.platform == "darwin":
        os.environ.setdefault("SDL_RENDER_DRIVER", "metal")
    return Viewer(args).run()


if __name__ == "__main__":
    sys.exit(main())
