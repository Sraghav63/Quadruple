#!/usr/bin/env python3
"""Screen build geometries with the same disturbed MuJoCo trials as robustness."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
from check_robustness import PUSH_FORCE, Trial, positive, run_trials

from quad_pendulum.linear import unstable_poles
from quad_pendulum.params import RigParams

# rho * pi/4 * (OD² - ID²), SI. Tube assumes a 1 mm wall (10 mm OD / 8 mm ID).
# Nominal bulk densities: aluminium 2700 kg/m³, steel 7850 kg/m³.
MATERIALS = {
    "10mm Al tube": 2700 * np.pi / 4 * (0.010**2 - 0.008**2),
    "12mm Al rod": 2700 * np.pi / 4 * 0.012**2,
    "10mm steel rod": 7850 * np.pi / 4 * 0.010**2,
    "12mm steel rod": 7850 * np.pi / 4 * 0.012**2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="16 candidates, one 4 s trial each")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seconds", type=positive, default=None)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (1 if args.quick else 3)
    seconds = args.seconds if args.seconds is not None else (4.0 if args.quick else 6.0)
    if args.workers < 1 or seeds < 1 or seconds < RigParams().control_dt:
        parser.error("workers/seeds must be positive; seconds must include a control tick")
    lengths = [0.25, 0.35] if args.quick else [0.20, 0.25, 0.30, 0.35, 0.40]
    frictions = [0.5, 2.0] if args.quick else [0.25, 0.5, 1.0, 2.0]
    candidates = []
    trials = []
    for length in lengths:
        for material, density in MATERIALS.items():
            for friction in frictions:
                rig = RigParams().with_links([length] * 4, [density * length] * 4)
                rig = replace(rig, joints=replace(rig.joints, coulomb=friction * 0.001))
                candidates.append((length, material, density, friction, rig))
                # Common seeds let candidates face identical tilts and push schedules.
                trials.extend(
                    Trial(2026 + seed, seconds, rig, perturbations=None) for seed in range(seeds)
                )
    results = run_trials(trials, args.workers)
    ranked = []
    for i, candidate in enumerate(candidates):
        batch = results[i * seeds : (i + 1) * seeds]
        rate = float(np.mean([r.survived for r in batch]))
        rail = max(r.rail_fraction for r in batch)
        force = max(r.max_force for r in batch)
        accel = max(r.max_accel for r in batch)
        poles = unstable_poles(candidate[-1])
        ranked.append((rate, rail, force, accel, poles, candidate))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    print(
        f"{len(candidates)} candidates; {seeds} seeds each; {seconds:g} s; "
        f"±{PUSH_FORCE:g} N / 20 ms Poisson pushes, 0.2 m reference at 3 s."
    )
    print("Ranked by survival, then peak rail use, then peak force. No parameter mismatch.")
    print(" #  length  material        kg/m  friction  pass   rail%    F(N)  a(m/s²)  poles(rad/s)")
    for i, (rate, rail, force, accel, poles, candidate) in enumerate(ranked, 1):
        length, material, density, friction, _ = candidate
        print(
            f"{i:2d}  {length:.2f} m  {material:14s} {density:.3f}  {friction:4.2f}mNm"
            f" {rate:5.0%} {rail:7.1%} {force:7.2f} {accel:8.2f}  "
            + ",".join(f"{p:.2f}" for p in poles)
        )
    winner = ranked[0]
    if winner[0] == 1.0:
        length, material, density, friction, _ = winner[-1]
        print(
            f"\nScreening recommendation: {length:.2f} m {material} ({density:.3f} kg/m), "
            f"joint Coulomb {friction:g} mN·m (tested), conditional on measured hardware matching."
        )
    else:
        print("\nNo candidate passed every seed: no build recommendation from this screen.")
    print(
        "Tube: 10 mm OD / 8 mm ID; rho_Al=2700, rho_steel=7850 kg/m³. "
        "Top-joint masses remain 60 g (tip 0 g)."
    )
    print("Small-sample screening only; re-run a full mismatch robustness study before building.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
