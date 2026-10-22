#!/usr/bin/env python
"""Plan and validate the transition library (needs `pip install casadi`).

For each preset rig, plans transitions between equilibria by direct collocation,
then checks each candidate closed loop in the MuJoCo plant (the real-time
supervisor: centre, track with time-varying LQR, hand off to the balancing
controller) from a few seeded tilts. Accepted transitions are written to
data/transitions/<preset>/<start>-<goal>.json with an index.json. Existing files
are kept unless --force, so the script can be resumed.

Default edges: swing-up from all-down to every configuration, swing-down from
every configuration to all-down, and every single-link flip. The runtime routes
through them, so every pair of configurations is connected.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from quad_pendulum.params import all_configurations, parse_configuration, wrap_angle
from quad_pendulum.presets import preset_rig, presets
from quad_pendulum.sim import Simulation
from quad_pendulum.transition import DATA_DIR, TransitionLibrary, Trajectory, plan_transition


def default_edges(n: int) -> list[tuple[str, str]]:
    down = "D" * n
    configs = all_configurations(n)
    edges = set()
    for c in configs:
        if c != down:
            edges.add((down, c))
            edges.add((c, down))
        for i in range(n):
            flipped = c[:i] + ("U" if c[i] == "D" else "D") + c[i + 1 :]
            edges.add((c, flipped))
    return sorted(edges)


def validate(preset: str, traj: Trajectory, seeds: int, tilt_deg: float) -> tuple[int, float]:
    """Closed-loop trials through Simulation; returns (successes, worst tracking deg)."""
    rig = preset_rig(preset)
    lib = TransitionLibrary(trajectories=[traj])
    ok, worst = 0, 0.0
    for seed in range(seeds):
        sim = Simulation(rig, seed=seed, configuration=traj.start, transitions=lib)
        sim.reset(random_tilt_deg=tilt_deg)
        sim.run(0.3)
        sim.request_configuration(traj.goal)
        end = sim.t + 2.0 + traj.duration + 2.0
        while sim.t < end and not sim.fallen:
            sim.step()
            if sim.mode == "transition":
                k = min(sim.tracker.k, sim.tracker.steps)
                theta = sim.plant.state()["theta"]
                dev = np.abs(wrap_angle(theta - sim.tracker.nominal[3 : 3 + rig.n_links, k]))
                worst = max(worst, float(dev.max()))
        ok += int(not sim.fallen and sim.mode == "hold" and sim.configuration == traj.goal)
    return ok, float(np.rad2deg(worst))


def plan_edge(args) -> dict:
    preset, start, goal, durations, attempts, seeds, tilt = args
    rig = preset_rig(preset)
    t0 = time.time()
    tried = 0
    best = None
    for duration in durations:
        for attempt in range(attempts):
            tried += 1
            traj = plan_transition(rig, start, goal, duration, seed=attempt, max_iter=1500)
            if traj is None:
                continue
            ok, worst = validate(preset, traj, seeds, tilt)
            if ok == seeds and (best is None or worst < best[1]):
                best = (traj, worst)
            if best is not None:
                break
        if best is not None:
            break
    result = {"preset": preset, "start": start, "goal": goal, "tried": tried,
              "seconds": round(time.time() - t0, 1), "ok": best is not None}
    if best is not None:
        traj, worst = best
        out = DATA_DIR / preset / f"{start}-{goal}.json"
        traj.save(out)
        result.update(duration=traj.duration, worst_deg=round(worst, 2), cost=round(traj.cost, 1))
    return result


def write_index(preset: str) -> int:
    folder = DATA_DIR / preset
    entries = []
    for f in sorted(folder.glob("*-*.json")):
        d = json.loads(f.read_text())
        entries.append({"start": d["start"], "goal": d["goal"], "duration": d["duration"],
                        "file": f.name})
    (folder / "index.json").write_text(json.dumps(entries, indent=1))
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", action="append", help="preset id (repeatable; default all)")
    ap.add_argument("--pairs", help="comma-separated START-GOAL list instead of the default edges")
    ap.add_argument("--attempts", type=int, default=4, help="random initial guesses per duration")
    ap.add_argument("--seeds", type=int, default=3, help="closed-loop validation trials")
    ap.add_argument("--tilt-deg", type=float, default=0.3, help="validation start tilt")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="replan edges that already exist")
    args = ap.parse_args()

    ids = args.preset or [p[0] for p in presets()]
    jobs = []
    for pid in ids:
        n = preset_rig(pid).n_links
        if args.pairs:
            edges = [tuple(parse_configuration(c, n) for c in p.split("-")) for p in args.pairs.split(",")]
        else:
            edges = default_edges(n)
        (DATA_DIR / pid).mkdir(parents=True, exist_ok=True)
        for start, goal in edges:
            if not args.force and (DATA_DIR / pid / f"{start}-{goal}.json").exists():
                continue
            flips = sum(a != b for a, b in zip(start, goal))
            base = 1.5 + 0.5 * flips
            durations = [base, base + 1.0, base + 2.0]
            jobs.append((pid, start, goal, durations, args.attempts, args.seeds, args.tilt_deg))
    print(f"{len(jobs)} transitions to plan on {args.workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(plan_edge, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            tag = f"{r['preset']:10s} {r['start']}->{r['goal']}"
            if r["ok"]:
                print(f"[{i}/{len(jobs)}] {tag} T={r['duration']:.1f}s track<={r['worst_deg']:.1f}deg "
                      f"({r['tried']} tries, {r['seconds']}s)", flush=True)
            else:
                failed.append(tag)
                print(f"[{i}/{len(jobs)}] {tag} FAILED ({r['tried']} tries, {r['seconds']}s)", flush=True)
    for pid in ids:
        print(f"{pid}: {write_index(pid)} transitions in {DATA_DIR / pid}")
        lib = TransitionLibrary(DATA_DIR / pid)
        n = preset_rig(pid).n_links
        missing = [(a, b) for a in all_configurations(n) for b in all_configurations(n)
                   if a != b and lib.route(a, b) is None]
        print(f"  unreachable pairs: {len(missing)}")
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    main()
