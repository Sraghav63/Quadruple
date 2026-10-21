#!/usr/bin/env python3
"""Reproducible nonlinear controller sweep; never changes the hardware model.

Example: .venv/bin/python scripts/tune_controller.py --suite rate --workers 4
Measured compromise points: --suite frontier; finer grid: --suite integral.
The final tracking-aware refinement is --suite tracking.
Random sweeps: --suite search --samples 240 --sweep-seed 431,
--suite weighted --sweep-seed 192, or --suite integral-search --samples 216
--sweep-seed 332. Short --seeds/--tap-seeds lists are screening only.
Use --candidates FILE for a JSON list of {"name": ..., "config": {overrides}}.
Each candidate gets fresh MuJoCo plants with identical seeds: 1 s settling,
4 s idle, then separate 20 ms top-link taps with 4 s recovery. Idle statistics
are worst-seed RMS/p99 and per-link absolute peaks, sampled at 1 kHz. Taps must
survive every seed and both directions; recovery means staying inside the
plant's fall/rail bounds for the entire recovery window, not a settling claim.
JSON output records the complete controller config and measurement protocol.
The existing 20 cm reference-step test is also measured (seed 2); passing
means the final 3 s average is within 8 cm after a 6 s response, without falling.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from importlib.metadata import version
from itertools import product
from pathlib import Path

# Must precede NumPy/SciPy imports, including in macOS spawn workers.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from quad_pendulum.controller import ControllerConfig, LQRController
from quad_pendulum.params import RigParams
from quad_pendulum.sim import Simulation


def candidates(suite: str, samples: int = 240, sweep_seed: int = 431) -> list[dict]:
    # Freeze the original baseline even after the production defaults change.
    baseline = asdict(
        ControllerConfig(
            cart_position_tol=0.15,
            cart_velocity_tol=1.0,
            angle_tol=0.2,
            rate_tol=1.0,
            accel_tol=3.0,
            joint_angle_tol=0.01,
            joint_rate_tol=None,
            joint_angle_weights=None,
            cart_accel_noise=0.5,
            link_accel_noise=2.0,
            command_delta_tol=None,
            joint_torque_noise=0.0,
            disturbance_time_constant=None,
            disturbance_torque_noise=0.0,
            design_model_lag=None,
            model_servo_dynamics=False,
            angle_integral_time_constant=None,
            angle_integral_tol=0.01,
        )
    )
    entries = [
        {"name": "interim", "config": baseline},
        {"name": "bryson", "config": {**baseline, "joint_angle_tol": None}},
        {"name": "chosen", "config": {}},
    ]
    if suite == "rate":
        for effort, delta in product((3.0, 10.0, 30.0), (0.03, 0.1, 0.3, 1.0)):
            entries.append(
                {
                    "name": f"rate-a{effort}-d{delta}",
                    "config": {**baseline, "accel_tol": effort, "command_delta_tol": delta},
                }
            )
    elif suite == "filter":
        for angular, torque in product((0.1, 0.5, 2.0), (0.0, 0.001, 0.01)):
            entries.append(
                {
                    "name": f"filter-w{angular}-t{torque}",
                    "config": {
                        **baseline,
                        "link_accel_noise": angular,
                        "joint_torque_noise": torque,
                    },
                }
            )
    elif suite == "disturbance":
        for tau, noise in product((0.02, 0.1, 0.5), (1e-5, 1e-4, 1e-3)):
            entries.append(
                {
                    "name": f"dist-t{tau}-w{noise}",
                    "config": {
                        **baseline,
                        "disturbance_time_constant": tau,
                        "disturbance_torque_noise": noise,
                    },
                }
            )
    elif suite == "frontier":
        for name, overrides in (
            (
                "quiet",
                {
                    "cart_position_tol": 0.08,
                    "joint_angle_tol": 0.02,
                    "link_accel_noise": 1.2,
                    "angle_integral_time_constant": 0.125,
                    "angle_integral_tol": 0.001,
                },
            ),
            (
                "tighter",
                {
                    "accel_tol": 1.1529978322350487,
                    "joint_angle_tol": 0.009769235515547545,
                    "cart_accel_noise": 0.212403683356804,
                    "link_accel_noise": 2.957690935570442,
                    "model_servo_dynamics": True,
                    "angle_integral_time_constant": 0.06411392160625984,
                    "angle_integral_tol": 0.0007001333620238328,
                },
            ),
        ):
            entries.append(
                {
                    "name": name,
                    "config": {
                        **baseline,
                        "accel_tol": 1.0,
                        "cart_accel_noise": 0.06,
                        "link_accel_noise": 1.0,
                        **overrides,
                    },
                }
            )
    elif suite == "tracking":
        for i, (cart, effort, noise, tau) in enumerate(
            product(
                (0.05, 0.065, 0.08),
                (0.8, 0.9, 1.0),
                (1.2, 1.4),
                (0.125, 0.15),
            )
        ):
            entries.append(
                {
                    "name": f"final-grid-{i}",
                    "config": {
                        **baseline,
                        "cart_position_tol": cart,
                        "accel_tol": effort,
                        "link_accel_noise": noise,
                        "joint_angle_tol": 0.02,
                        "cart_accel_noise": 0.06,
                        "angle_integral_time_constant": tau,
                        "angle_integral_tol": 0.001,
                    },
                }
            )
    elif suite == "integral":
        for i, (joint, itol, tau, effort, noise) in enumerate(
            product(
                (0.012, 0.02, 0.03),
                (0.0006, 0.001, 0.0016),
                (0.1, 0.15, 0.25),
                (0.7, 1.0, 1.4),
                (1.0, 1.5, 2.0),
            )
        ):
            entries.append(
                {
                    "name": f"integral-grid-{i}",
                    "config": {
                        **baseline,
                        "joint_angle_tol": joint,
                        "angle_integral_tol": itol,
                        "angle_integral_time_constant": tau,
                        "accel_tol": effort,
                        "link_accel_noise": noise,
                        "cart_accel_noise": 0.06,
                    },
                }
            )
    elif suite in ("search", "weighted", "integral-search"):
        rng = np.random.default_rng(sweep_seed)

        def log(low, high):
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))

        for i in range(samples):
            if suite == "search":
                cfg = {
                    **baseline,
                    "accel_tol": log(1, 20),
                    "joint_angle_tol": log(0.001, 0.06),
                    "rate_tol": log(0.2, 5),
                    "command_delta_tol": log(0.03, 5) if rng.random() < 0.8 else None,
                    "link_accel_noise": log(0.1, 4),
                    "cart_accel_noise": log(0.03, 0.8),
                    "design_model_lag": log(0.001, 0.01),
                    "cart_position_tol": log(0.05, 0.3),
                }
                if i >= 2 * samples // 3:
                    cfg["joint_torque_noise"] = log(0.0001, 0.02)
            elif suite == "weighted":
                cfg = {
                    **baseline,
                    "accel_tol": log(0.7, 6),
                    "joint_angle_weights": [
                        log(0.1, 20),
                        log(0.03, 10),
                        log(0.01, 5),
                        log(0.005, 5),
                    ],
                    "command_delta_tol": log(0.1, 2) if rng.random() < 0.8 else None,
                    "link_accel_noise": log(0.3, 3),
                    "cart_accel_noise": log(0.05, 0.8),
                }
            else:
                cfg = {
                    **baseline,
                    "accel_tol": log(0.7, 6),
                    "joint_angle_tol": log(0.005, 0.04),
                    "command_delta_tol": log(0.1, 2) if rng.random() < 0.5 else None,
                    "link_accel_noise": log(0.1, 3),
                    "cart_accel_noise": log(0.03, 0.8),
                    "angle_integral_time_constant": log(0.05, 2),
                    "angle_integral_tol": log(0.0001, 0.02),
                    "model_servo_dynamics": i >= 2 * samples // 3,
                }
            entries.append({"name": f"{suite}-{i}", "config": cfg})
    return entries


def make_sim(config: ControllerConfig, seed: int) -> Simulation:
    sim = Simulation(seed=seed)
    sim.controller = LQRController(RigParams(), config)
    sim.reset(random_tilt_deg=0.05)
    return sim


def advance(sim: Simulation, seconds: float) -> bool:
    for _ in range(round(seconds / sim.rig.control_dt)):
        sim.step()
        if sim.fallen:
            return False
    return True


def evaluate(job: tuple[dict, dict]) -> dict:
    entry, protocol = job
    result = {"name": entry["name"], "config": entry["config"]}
    try:
        cfg = replace(ControllerConfig(), **entry["config"])
        result["config"] = asdict(cfg)
        idle = []
        for seed in protocol["seeds"]:
            sim = make_sim(cfg, seed)
            survived = advance(sim, protocol["settle_seconds"])
            commands, angles, positions = [], [], []
            if survived:
                for _ in range(round(protocol["idle_seconds"] / sim.rig.control_dt)):
                    # This is the newly calculated command, before transport delay.
                    sim.step()
                    commands.append(
                        float(sim.controller.z[-1]) if sim.controller.delay else sim.last_command
                    )
                    state = sim.plant.state()
                    angles.append(np.rad2deg(np.abs(state["theta"])))
                    positions.append(abs(state["x"]))
                    if sim.fallen:
                        survived = False
                        break
            idle.append(
                {
                    "seed": seed,
                    "survived": survived,
                    "rms": float(np.sqrt(np.mean(np.square(commands)))) if commands else None,
                    "p99": float(np.percentile(np.abs(commands), 99)) if commands else None,
                    "max_theta_deg": np.max(angles, axis=0).tolist() if angles else None,
                    "max_x": max(positions) if positions else None,
                }
            )
        taps = []
        for force in protocol["forces"]:
            for seed, direction in product(protocol["tap_seeds"], (-1, 1)):
                sim = make_sim(cfg, seed)
                survived = advance(sim, protocol["settle_seconds"])
                if survived:
                    sim.push(3, direction * force, 0.020)
                    survived = advance(sim, protocol["recovery_seconds"])
                taps.append(
                    {
                        "force": direction * force,
                        "seed": seed,
                        "survived": survived,
                        "end_time": sim.t,
                    }
                )
        result.update(idle=idle, taps=taps)
        result["idle_survived"] = all(r["survived"] for r in idle)
        if all(r["rms"] is not None for r in idle):
            result["rms"] = max(r["rms"] for r in idle)
            result["p99"] = max(r["p99"] for r in idle)
            result["max_theta_deg"] = np.max([r["max_theta_deg"] for r in idle], axis=0).tolist()
            result["theta"] = max(result["max_theta_deg"])
        result["tap_pass"] = sum(t["survived"] for t in taps)
        result["tap_total"] = len(taps)
        result["push"] = max(
            (
                f
                for f in protocol["forces"]
                if all(t["survived"] for t in taps if abs(t["force"]) <= f)
            ),
            default=0.0,
        )
        if protocol.get("tracking", True):
            sim = make_sim(cfg, 2)
            survived = advance(sim, 1.0)
            sim.set_reference(0.2)
            positions = []
            if survived:
                for _ in range(round(6.0 / sim.rig.control_dt)):
                    sim.step()
                    positions.append(sim.plant.state()["x"])
                    if sim.fallen:
                        survived = False
                        break
            mean_x = (
                float(np.mean(positions[-round(3.0 / sim.rig.control_dt) :])) if positions else None
            )
            result["tracking_mean_x"] = mean_x
            result["tracking_pass"] = bool(survived and abs(mean_x - 0.2) < 0.08)
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["error"] = str(exc)
    return result


def pareto(results: list[dict]) -> list[dict]:
    valid = [
        r
        for r in results
        if r.get("idle_survived")
        and "rms" in r
        and "error" not in r
        and r.get("tracking_pass", True)
    ]

    def objectives(r):
        return np.array([r["rms"], r["p99"], r["theta"], -r["push"]])

    return [
        r
        for r in valid
        if not any(
            np.all(objectives(other) <= objectives(r)) and np.any(objectives(other) < objectives(r))
            for other in valid
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "baseline",
            "rate",
            "filter",
            "disturbance",
            "integral",
            "search",
            "weighted",
            "integral-search",
            "frontier",
            "tracking",
        ),
        default="baseline",
    )
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--samples", type=int, default=240, help="size of random suites")
    parser.add_argument(
        "--sweep-seed", type=int, default=431, help="random candidate generator seed"
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--tap-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--forces", type=float, nargs="*", default=[0.08, 0.12, 0.16])
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--idle-seconds", type=float, default=4.0)
    parser.add_argument("--recovery-seconds", type=float, default=4.0)
    parser.add_argument("--skip-tracking", action="store_true", help="screening only")
    parser.add_argument("--output", type=Path, default=Path("/tmp/quad-controller-sweep.json"))
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.samples < 1
        or args.sweep_seed < 0
        or any(seed < 0 for seed in args.seeds + args.tap_seeds)
    ):
        parser.error("workers must be positive and seeds nonnegative")
    durations = (args.settle_seconds, args.idle_seconds, args.recovery_seconds)
    if any(not np.isfinite(t) or t <= 0 for t in durations):
        parser.error("durations must be finite and positive")
    if any(round(t / RigParams().control_dt) < 1 for t in durations):
        parser.error("durations must include at least one control tick")
    if any(not np.isfinite(f) or f <= 0 for f in args.forces):
        parser.error("forces must be finite and positive")
    if args.candidates:
        entries = json.loads(args.candidates.read_text())
        if isinstance(entries, dict):
            entries = entries["results"]  # directly replay a previous output file
    else:
        entries = candidates(args.suite, args.samples, args.sweep_seed)
    if not entries:
        parser.error("at least one candidate is required")
    protocol = {
        name: getattr(args, name)
        for name in (
            "seeds",
            "tap_seeds",
            "forces",
            "settle_seconds",
            "idle_seconds",
            "recovery_seconds",
        )
    }
    protocol["tracking"] = not args.skip_tracking
    metadata = {
        "rig": asdict(RigParams()),
        "protocol": protocol,
        "generator": {
            "suite": args.suite,
            "samples": args.samples,
            "seed": args.sweep_seed,
            "source": str(args.candidates) if args.candidates else None,
        },
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "mujoco")},
        },
    }
    jobs = [(entry, protocol) for entry in entries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.workers == 1:
        iterator = map(evaluate, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
        iterator = pool.map(evaluate, jobs)
    try:
        for r in iterator:
            results.append(r)
            if "rms" in r:
                tracking = "skipped"
                if "tracking_pass" in r:
                    tracking = f"{'PASS' if r['tracking_pass'] else 'FAIL'}"
                    if r["tracking_mean_x"] is not None:
                        tracking += f"(mean_x={r['tracking_mean_x']:.3f})"
                print(
                    f"{r['name']:32s} rms={r['rms']:6.3f} p99={r['p99']:6.3f} "
                    f"theta={r['theta']:6.3f} tap={r['tap_pass']}/{r['tap_total']} "
                    f"push={r['push']:.3f} idle={r['idle_survived']} tracking={tracking}",
                    flush=True,
                )
            else:
                print(f"{r['name']}: {r.get('error', 'fell during settle')}", flush=True)
            args.output.write_text(json.dumps({**metadata, "results": results}, indent=2) + "\n")
    finally:
        if pool:
            pool.shutdown()
    print("Pareto:", ", ".join(r["name"] for r in pareto(results)))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
