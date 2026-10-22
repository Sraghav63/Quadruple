#!/usr/bin/env python3
"""Seeded model-mismatch trials; all peaks are sampled at the control rate."""

from __future__ import annotations

import argparse
import functools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

# Each process runs small matrix solves; threaded BLAS would oversubscribe CPUs.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from quad_pendulum.params import (
    RigParams,
    all_configurations,
    parse_configuration,
    wrap_angle,
)
from quad_pendulum.presets import preset_rig
from quad_pendulum.sim import Simulation
from quad_pendulum.transition import DATA_DIR, Trajectory, TransitionLibrary

TRANSITION_HOLD_SECONDS = 6.0  # s balancing at START before the request

# Random taps: a quadruple pendulum on a 30 m/s^2 cart only survives light ones.
PUSH_FORCE = 0.08  # N for 20 ms (1.6 mN s), on a random link
HOLD_AFTER_ARRIVAL = 3.0  # s a transition must keep balancing at the goal to count


@dataclass(frozen=True)
class Perturbations:
    length: float = 1.0
    mass: float = 1.0
    cart_mass: float = 1.0
    friction: float = 1.0
    viscous: float = 1.0
    bandwidth: float = 1.0
    delay: float = 1.0


@dataclass(frozen=True)
class Trial:
    seed: int
    seconds: float = 8.0
    configuration: str | None = None
    nominal: RigParams = field(default_factory=RigParams)
    perturbations: Perturbations | None = field(default_factory=Perturbations)
    push_scale: float = 1.0


@dataclass(frozen=True)
class Result:
    seed: int
    survived: bool
    fall_time: float | None
    max_x: float
    max_accel: float
    max_force: float
    rms_angles_deg: tuple[float, ...]
    rail_fraction: float


def perturb_rig(rig: RigParams, rng: np.random.Generator, p: Perturbations) -> RigParams:
    """Scales multiply uncertainty widths; zero disables that uncertainty."""

    def factor(low: float, high: float, scale: float) -> float:
        return 1.0 + scale * (rng.uniform(low, high) - 1.0)

    links = tuple(
        replace(
            link,
            length=link.length * factor(0.98, 1.02, p.length),
            rod_mass=link.rod_mass * factor(0.9, 1.1, p.mass),
            top_mass=link.top_mass * factor(0.9, 1.1, p.mass),
        )
        for link in rig.links
    )
    return replace(
        rig,
        links=links,
        cart=replace(rig.cart, mass=rig.cart.mass * factor(0.9, 1.1, p.cart_mass)),
        joints=replace(
            rig.joints,
            coulomb=rig.joints.coulomb * factor(0.5, 2.0, p.friction),
            viscous=rig.joints.viscous * factor(0.5, 2.0, p.viscous),
        ),
        drive=replace(
            rig.drive,
            velocity_bandwidth=rig.drive.velocity_bandwidth * factor(0.6, 1.4, p.bandwidth),
        ),
        timing=replace(
            rig.timing,
            compute_delay_steps=rig.timing.compute_delay_steps
            + int(rng.random() < min(1.0, 0.2 * p.delay)),
        ),
    )


def run_trial(trial: Trial) -> Result:
    parameter_seed, event_seed = np.random.SeedSequence(trial.seed).spawn(2)
    rng = np.random.default_rng(event_seed)
    true_rig = (
        trial.nominal
        if trial.perturbations is None
        else perturb_rig(trial.nominal, np.random.default_rng(parameter_seed), trial.perturbations)
    )
    # True plant and nominal controller share the target equilibrium; reset tilts about it.
    sim = Simulation(
        rig=true_rig,
        controller_rig=trial.nominal,
        seed=trial.seed,
        configuration=trial.configuration,
    )
    sim.reset(random_tilt_deg=0.05)
    next_push = rng.exponential(2.0)
    max_x = max_accel = max_force = 0.0
    angle_squares = np.zeros(true_rig.n_links)
    count = 0
    for _ in range(round(trial.seconds / true_rig.control_dt)):
        if sim.t + 1e-10 >= 3.0:
            sim.set_reference(0.2)
        while sim.t + 1e-10 >= next_push:
            sim.push(
                int(rng.integers(true_rig.n_links)),
                float(rng.choice([-1.0, 1.0])) * PUSH_FORCE * trial.push_scale,
                0.020,
            )
            next_push += rng.exponential(2.0)
        sim.step()
        state = sim.plant.state()
        values = np.r_[state["x"], state["a"], state["force"], state["theta"]]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite plant state in seed {trial.seed} at {sim.t}")
        max_x = max(max_x, abs(state["x"]))
        max_accel = max(max_accel, abs(state["a"]))
        max_force = max(max_force, abs(state["force"]))
        angle_squares += wrap_angle(np.asarray(state["theta"]) - sim.theta_eq) ** 2
        count += 1
        if sim.fallen:
            break
    return Result(
        trial.seed,
        not sim.fallen,
        sim.t if sim.fallen else None,
        max_x,
        max_accel,
        max_force,
        tuple(np.rad2deg(np.sqrt(angle_squares / count))),
        max_x / true_rig.cart.rail_half_length,
    )


@dataclass(frozen=True)
class TransitionTrial:
    seed: int
    start: str
    goal: str
    seconds: float = 40.0
    preset: str = "quad"
    nominal: RigParams = field(default_factory=RigParams)
    perturbations: Perturbations | None = field(default_factory=Perturbations)
    push_scale: float = 0.0


@dataclass(frozen=True)
class TransitionResult:
    seed: int
    start: str
    goal: str
    success: bool
    arrival_time: float | None
    fall_time: float | None
    worst_deviation_deg: float
    reason: str


@functools.cache
def load_library(preset: str) -> TransitionLibrary | None:
    directory = DATA_DIR / preset
    files = sorted(directory.glob("*-*.json")) if directory.is_dir() else []
    if not files:
        return None
    if (directory / "index.json").exists():
        return TransitionLibrary(directory)
    return TransitionLibrary(trajectories=[Trajectory.load(f) for f in files])


def run_transition_trial(trial: TransitionTrial) -> TransitionResult:
    """Start at `start` (at rest, tiny tilt), request `goal`, then keep balancing there."""
    parameter_seed, event_seed = np.random.SeedSequence(trial.seed).spawn(2)
    rng = np.random.default_rng(event_seed)
    true_rig = (
        trial.nominal
        if trial.perturbations is None
        else perturb_rig(trial.nominal, np.random.default_rng(parameter_seed), trial.perturbations)
    )
    sim = Simulation(
        rig=true_rig,
        controller_rig=trial.nominal,
        seed=trial.seed,
        configuration=trial.start,
        transitions=load_library(trial.preset),
    )
    sim.reset(random_tilt_deg=0.05)

    def result(ok: bool, reason: str, arrival=None, worst=0.0) -> TransitionResult:
        fall = sim.t if sim.fallen else None
        return TransitionResult(trial.seed, trial.start, trial.goal, ok, arrival, fall, worst, reason)

    # Balance first: joint stiction keeps the chain wobbling, and the swing has
    # to start from that, as it would on the rig.
    sim.run(TRANSITION_HOLD_SECONDS)
    if sim.fallen:
        return result(False, "fell before the request")
    if not sim.request_configuration(trial.goal):
        return result(False, "no route")
    next_push = sim.t + rng.exponential(2.0) if trial.push_scale > 0 else np.inf
    arrival = None
    requested_at = sim.t
    worst = 0.0
    n = true_rig.n_links
    for _ in range(round(trial.seconds / true_rig.control_dt)):
        while sim.t + 1e-10 >= next_push:
            sim.push(
                int(rng.integers(n)),
                float(rng.choice([-1.0, 1.0])) * PUSH_FORCE * trial.push_scale,
                0.020,
            )
            next_push += rng.exponential(2.0)
        sim.step()
        if sim.mode == "transition":
            tr = sim.tracker
            nominal = tr.nominal[3 : 3 + n, min(tr.k, tr.steps)]
            theta = np.asarray(sim.plant.state()["theta"])
            worst = max(worst, float(np.rad2deg(np.abs(wrap_angle(theta - nominal)).max())))
        if not np.isfinite(sim.plant.state()["x"]):
            raise RuntimeError(f"Non-finite plant state in seed {trial.seed} at {sim.t}")
        if sim.fallen:
            return result(False, "fell" if arrival is None else "fell after arrival", arrival, worst)
        if arrival is None and sim.mode == "hold" and not sim.route:
            arrival = sim.t - requested_at  # time from the request
        if arrival is not None and sim.t - requested_at >= arrival + HOLD_AFTER_ARRIVAL:
            return result(True, "ok", arrival, worst)
    return result(False, "timeout" if arrival is None else "ended early", arrival, worst)


def run_transition_trials(trials: list[TransitionTrial], workers: int) -> list[TransitionResult]:
    if workers == 1:
        return [run_transition_trial(trial) for trial in trials]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_transition_trial, trials))


def run_trials(trials: list[Trial], workers: int) -> list[Result]:
    if workers == 1:
        return [run_trial(trial) for trial in trials]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_trial, trials))


def positive(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def transition_report(args, parser, nominal: RigParams, p: Perturbations) -> int:
    library = load_library(args.preset)
    if library is None:
        parser.error(
            f"no transition library for preset {args.preset!r} in {DATA_DIR / args.preset}; "
            "generate it with scripts/plan_transitions.py"
        )
    edges = list(library.edges) if args.all_transitions else []
    for spec in args.transition:
        try:
            start, goal = (parse_configuration(c, nominal.n_links) for c in spec.split("-"))
        except ValueError:
            parser.error(f"--transition expects START-GOAL, e.g. DD-UU (got {spec!r})")
        if (start, goal) not in edges:
            edges.append((start, goal))
    trials = [
        TransitionTrial(
            args.seed + i,
            start,
            goal,
            args.seconds,
            args.preset,
            nominal,
            None if args.nominal else p,
            args.push_scale,
        )
        for start, goal in edges
        for i in range(args.trials)
    ]
    results = run_transition_trials(trials, args.workers)
    print(
        f"Preset {args.preset}: start at START, request GOAL, success = arrives in hold at GOAL "
        f"and balances {HOLD_AFTER_ARRIVAL:.0f} s more (taps scale {args.push_scale:g})."
    )
    print("edge             success  rate    median arrival(s)  worst dev(deg)  failures")
    worst_rate = 1.0
    for k, (start, goal) in enumerate(edges):
        rs = results[k * args.trials : (k + 1) * args.trials]
        rate = float(np.mean([r.success for r in rs]))
        worst_rate = min(worst_rate, rate)
        arrivals = [r.arrival_time for r in rs if r.arrival_time is not None]
        median = f"{np.median(arrivals):.2f}" if arrivals else "-"
        reasons: dict[str, int] = {}
        for r in rs:
            if not r.success:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        fails = ", ".join(f"{v} {k}" for k, v in reasons.items()) or "-"
        print(
            f"{start}-{goal:<{max(1, 12 - len(start))}s}  {sum(r.success for r in rs):3d}/{len(rs):<3d}"
            f"  {rate:6.1%}  {median:>17s}  {max(r.worst_deviation_deg for r in rs):14.2f}  {fails}"
        )
    print(f"\nWorst edge success {worst_rate:.1%}; required {args.min_success:.1%}")
    return int(worst_rate < args.min_success)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--seconds",
        type=positive,
        default=None,
        help="trial length (default 8; 40 for transitions, including the hold after arrival)",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-success", type=float, default=0.9)
    parser.add_argument(
        "--nominal", action="store_true", help="no parameter mismatch; pushes remain"
    )
    parser.add_argument(
        "--push-scale",
        type=float,
        default=None,
        help="scale of random taps (default 1; 0 for transitions, which are tap-free unless set)",
    )
    parser.add_argument("--preset", default="quad", help="rig preset (default quad)")
    parser.add_argument(
        "--transition",
        action="append",
        default=[],
        metavar="START-GOAL",
        help="Monte Carlo of a planned transition, e.g. DDDD-UUUU (repeatable)",
    )
    parser.add_argument(
        "--all-transitions",
        action="store_true",
        help="every transition stored for the preset, with a per-edge table",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config", metavar="CFG", help="up/down configuration, bottom link first (default all U)"
    )
    group.add_argument(
        "--all-configs",
        action="store_true",
        help="run every U/D configuration and print a per-configuration survival table",
    )
    for name in Perturbations.__dataclass_fields__:
        parser.add_argument(
            f"--{name.replace('_', '-')}-scale",
            type=float,
            default=1.0,
            help="multiply uncertainty width (delay: extra-step probability / 0.2)",
        )
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1 or args.seed < 0:
        parser.error("trials/workers must be positive and seed nonnegative")
    if not np.isfinite(args.min_success) or not 0 <= args.min_success <= 1:
        parser.error("min-success must be in [0, 1]")
    p = Perturbations(
        **{name: getattr(args, name + "_scale") for name in Perturbations.__dataclass_fields__}
    )
    bounds = {
        "length": 50.0,
        "mass": 10.0,
        "cart_mass": 10.0,
        "friction": 2.0,
        "viscous": 2.0,
        "bandwidth": 2.5,
        "delay": float("inf"),
    }
    for name, upper in bounds.items():
        value = getattr(p, name)
        if not np.isfinite(value) or value < 0 or value >= upper:
            parser.error(f"{name}-scale must be finite, >= 0 and < {upper}")
    transitions_mode = bool(args.transition) or args.all_transitions
    if args.push_scale is None:
        args.push_scale = 0.0 if transitions_mode else 1.0
    if args.seconds is None:
        args.seconds = 40.0 if transitions_mode else 8.0
    if not np.isfinite(args.push_scale) or args.push_scale < 0:
        parser.error("push-scale must be finite and nonnegative")
    try:
        nominal = preset_rig(args.preset)
    except KeyError as exc:
        parser.error(str(exc))
    if args.seconds < nominal.control_dt:
        parser.error("seconds must include at least one control tick")
    n_links = nominal.n_links
    if transitions_mode:
        return transition_report(args, parser, nominal, p)
    try:
        configuration = parse_configuration(args.config, n_links)
    except ValueError as exc:
        parser.error(str(exc))
    configurations = all_configurations(n_links) if args.all_configs else [configuration]

    def make_trials(cfg: str) -> list[Trial]:
        return [
            Trial(
                args.seed + i,
                args.seconds,
                cfg,
                nominal=nominal,
                perturbations=None if args.nominal else p,
                push_scale=args.push_scale,
            )
            for i in range(args.trials)
        ]

    if args.all_configs:
        everything = run_trials([t for cfg in configurations for t in make_trials(cfg)], args.workers)
        print("config  survived  rate    median fall(s)  p95 rail use  p95 |F|(N)  max RMS angle(deg)")
        worst = 1.0
        for k, cfg in enumerate(configurations):
            rs = everything[k * args.trials : (k + 1) * args.trials]
            rate = float(np.mean([r.survived for r in rs]))
            worst = min(worst, rate)
            falls = [r.fall_time for r in rs if r.fall_time is not None]
            median_fall = f"{np.median(falls):.2f}" if falls else "-"
            print(
                f"{cfg:6s}  {sum(r.survived for r in rs):3d}/{len(rs):<3d}  {rate:6.1%}"
                f"  {median_fall:>14s}  {np.percentile([r.rail_fraction for r in rs], 95):11.1%}"
                f"  {np.percentile([r.max_force for r in rs], 95):10.2f}"
                f"  {max(max(r.rms_angles_deg) for r in rs):18.3f}"
            )
        print(f"\nWorst configuration survival {worst:.1%}; required {args.min_success:.1%}")
        return int(worst < args.min_success)
    print(f"Configuration {configuration} (bottom link first, U up / D down)")
    results = run_trials(make_trials(configuration), args.workers)
    print(" seed  result  fall(s)  max|x|(m)  max|a|(m/s²)  max|F|(N)  RMS angle error 1..4 (deg)")
    for r in results:
        fall = "-" if r.fall_time is None else f"{r.fall_time:.3f}"
        rms = " ".join(f"{angle:.3f}" for angle in r.rms_angles_deg)
        print(
            f"{r.seed:5d}  {'PASS' if r.survived else 'FALL':6s} {fall:>7s}"
            f" {r.max_x:10.3f} {r.max_accel:13.2f} {r.max_force:11.2f}  {rms}"
        )
    rate = np.mean([r.survived for r in results])
    print(
        f"\nSuccess: {sum(r.survived for r in results)}/{len(results)} ({rate:.1%}); "
        f"required {args.min_success:.1%}"
    )
    print("All trials, including failures; peaks sampled at 1 kHz; RMS until fall/end.")
    print(
        f"p95 rail use: {np.percentile([r.rail_fraction for r in results], 95):.1%} "
        f"of half-rail; p95 |force|: {np.percentile([r.max_force for r in results], 95):.2f} N"
    )
    print(
        f"Peak |x|: {max(r.max_x for r in results):.3f} m; "
        f"|accel|: {max(r.max_accel for r in results):.2f} m/s²; "
        f"|force|: {max(r.max_force for r in results):.2f} N"
    )
    return int(rate < args.min_success)


if __name__ == "__main__":
    raise SystemExit(main())
