#!/usr/bin/env python
"""Plan and validate the transition library (needs `pip install casadi`).

For each preset rig, plans transitions between equilibria by direct collocation,
then checks each candidate closed loop in the MuJoCo plant (the real-time
supervisor: centre, track with time-varying LQR, hand off to the balancing
controller) from a few seeded tilts. Accepted transitions are written to
data/transitions/<preset>/<start>-<goal>.json with an index.json. Existing files
are kept unless --force (or re-checked with --revalidate), so the script can be resumed.

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

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from quad_pendulum.params import all_configurations, parse_configuration, wrap_angle
from quad_pendulum.presets import preset_rig, presets
from quad_pendulum.sim import Simulation
from quad_pendulum.transition import (
    DATA_DIR,
    PlanLimits,
    Trajectory,
    TransitionLibrary,
    plan_transition,
    reversed_trajectory,
)


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


# Gentler plans leave the feedback more of the drive's 30 m/s^2 and 2.5 m/s;
# some transitions only track reliably that way, others only fit the rail fast.
LIMIT_PROFILES = (PlanLimits(), PlanLimits(rail=0.45, velocity=1.8, accel_command=18.0, force=100.0))
HOLD_SECONDS = 6.0  # balance first, so the transition starts from the stiction wobble


def validate(preset: str, traj: Trajectory, seeds: int, tilt_deg: float) -> tuple[int, float]:
    """Closed-loop trials through Simulation; returns (successes, worst tracking deg).

    Each trial balances the start configuration for HOLD_SECONDS (joint stiction
    keeps it wobbling, as it would on the rig), then requests the goal and must
    arrive in hold and stay up for 2 s."""
    rig = preset_rig(preset)
    lib = TransitionLibrary(trajectories=[traj])
    ok, worst = 0, 0.0
    for seed in range(seeds):
        sim = Simulation(rig, seed=seed, configuration=traj.start, transitions=lib)
        sim.reset(random_tilt_deg=tilt_deg)
        sim.run(HOLD_SECONDS)
        if sim.fallen or not sim.request_configuration(traj.goal):
            continue
        deadline = sim.t + 3.0 + traj.duration  # centring allowance + the swing
        arrived = None
        while not sim.fallen:
            sim.step()
            if arrived is None and sim.mode == "hold" and sim.configuration == traj.goal:
                arrived = sim.t
            if arrived is not None and sim.t >= arrived + 2.0:  # balance 2 s at the goal
                break
            if arrived is None and sim.t >= deadline:
                break
            if sim.mode == "transition":
                k = min(sim.tracker.k, sim.tracker.steps)
                theta = sim.plant.state()["theta"]
                dev = np.abs(wrap_angle(theta - sim.tracker.nominal[3 : 3 + rig.n_links, k]))
                worst = max(worst, float(dev.max()))
        ok += int(arrived is not None and not sim.fallen and sim.mode == "hold")
        if ok <= seed:  # one failure is enough to reject
            break
    return ok, float(np.rad2deg(worst))


def candidates(preset, rig, start, goal, flips, attempts):
    """Plans to try, cheapest first: each limit profile and duration, the reversed
    opposite transition as a guess (if planned), then random guesses."""
    reverse = DATA_DIR / preset / f"{goal}-{start}.json"
    for limits in LIMIT_PROFILES:
        gentle = limits.accel_command < PlanLimits().accel_command
        base = 1.5 + 0.5 * flips + (1.0 if gentle else 0.0)
        for duration in (base, base + 1.0, base + 2.0):
            if reverse.exists():
                rev = reversed_trajectory(Trajectory.load(reverse))
                yield plan_transition(rig, start, goal, duration, limits=limits, initial=rev, max_iter=1500)
            for attempt in range(attempts):
                yield plan_transition(rig, start, goal, duration, limits=limits, seed=attempt, max_iter=1500)


def plan_edge(args) -> dict:
    preset, start, goal, attempts, seeds, tilt, revalidate = args
    rig = preset_rig(preset)
    out = DATA_DIR / preset / f"{start}-{goal}.json"
    t0 = time.time()
    tried = 0
    found = None
    if out.exists():
        traj = Trajectory.load(out)
        ok, worst = validate(preset, traj, seeds, tilt) if revalidate else (seeds, 0.0)
        if ok == seeds:
            found = (traj, worst, "kept")
    if found is None:
        flips = sum(a != b for a, b in zip(start, goal))
        for traj in candidates(preset, rig, start, goal, flips, attempts):
            tried += 1
            if traj is None:
                continue
            ok, worst = validate(preset, traj, seeds, tilt)
            if ok == seeds:
                found = (traj, worst, "planned")
                break
    result = {"preset": preset, "start": start, "goal": goal, "tried": tried,
              "seconds": round(time.time() - t0, 1), "ok": found is not None}
    if found is not None:
        traj, worst, how = found
        traj.save(out)
        result.update(duration=traj.duration, worst_deg=round(worst, 2), how=how,
                      accel=traj.limits.accel_command)
    elif out.exists():
        out.unlink()  # an existing transition that no longer validates is dropped
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
    ap.add_argument("--attempts", type=int, default=2, help="random initial guesses per duration")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check existing transitions (dropped and replanned if they fail)")
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
            exists = (DATA_DIR / pid / f"{start}-{goal}.json").exists()
            if exists and args.force:
                (DATA_DIR / pid / f"{start}-{goal}.json").unlink()
            elif exists and not args.revalidate:
                continue
            jobs.append((pid, start, goal, args.attempts, args.seeds, args.tilt_deg, args.revalidate))
    print(f"{len(jobs)} transitions to plan on {args.workers} workers", flush=True)
    failed = []
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(plan_edge, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            tag = f"{r['preset']:10s} {r['start']}->{r['goal']}"
            if r["ok"]:
                print(f"[{i}/{len(jobs)}] {tag} {r['how']} T={r['duration']:.1f}s "
                      f"a<={r['accel']:.0f} track<={r['worst_deg']:.1f}deg "
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
