#!/usr/bin/env python
"""Export rig presets, controller matrices and reference traces for the web app.

Writes:
  web/public/presets/index.json           list of presets
  web/public/presets/<id>.json            rig parameters + controller matrices
  web/test/golden/<id>.controller.json    measurement -> command trace (parity test)
  web/test/golden/<id>.plant.json         MuJoCo open-loop trace (physics parity test)

The browser runs the same runtime as the firmware:
  reset:  z = 0; z[0] = y[0]; z[3:3+n] = y[1:]
  ref:    z[0] -= x_ref - previous_x_ref
  step:   z += L (y - C z); u = clamp(-K z); z = A z + B u
with y = [cart_position - x_ref, cumulative sum of the joint encoder angles].
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from quad_pendulum.controller import LQRController
from quad_pendulum.linear import unstable_poles
from quad_pendulum.params import Measurement, RigParams, absolute_to_relative
from quad_pendulum.plant import Plant

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "web" / "public" / "presets"
GOLDEN = ROOT / "web" / "test" / "golden"


def presets() -> list[tuple[str, str, str, RigParams]]:
    quad = RigParams()
    link = quad.links[0]
    out = [
        ("quad", "Quadruple (4 links)", "The build target: 4 x 0.35 m, 12 mm steel rod.", quad),
    ]
    for n, name in [(3, "Triple"), (2, "Double"), (1, "Single")]:
        rig = quad.with_links([link.length] * n, [link.rod_mass] * n, top_mass=link.top_mass)
        out.append((f"links{n}", f"{name} ({n} link{'s' if n > 1 else ''})", "Same links, fewer of them.", rig))
    short = quad.with_links([0.30] * 4, [0.27] * 4, top_mass=link.top_mass)
    out.append(
        ("quad-short", "Quadruple, 0.30 m links", "Shorter, faster, less robust (85% vs ~97% in Monte Carlo).", short)
    )
    return out


def camel(d):
    if isinstance(d, dict):
        return {_camel(k): camel(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [camel(v) for v in d]
    return d


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(word.capitalize() for word in rest)


def rig_json(rig: RigParams) -> dict:
    return camel(asdict(rig))


def controller_json(ctl: LQRController) -> dict:
    return {
        "nz": int(ctl.nz),
        "ny": int(1 + ctl.n),
        "dt": float(ctl.dt),
        "accelLimit": float(ctl.rig.drive.accel_limit),
        "thetaIndex": 3,
        "stateLayout": str(getattr(ctl, "state_layout", "")),
        "K": ctl.k[0].tolist(),
        "L": ctl.l.tolist(),
        "A": ctl.az.tolist(),
        "B": ctl.bz[:, 0].tolist(),
        "C": ctl.c.tolist(),
        "closedLoopRadius": float(ctl.closed_loop_radius),
        "observerRadius": float(ctl.observer_radius),
    }


def controller_golden(rig: RigParams, ctl: LQRController, steps: int = 400) -> dict:
    """Feed a deterministic measurement sequence; record Python's commands."""
    rng = np.random.default_rng(1)
    n = rig.n_links
    t = np.arange(steps) * rig.control_dt
    meas = []
    for k in range(steps):
        theta = 0.004 * np.sin(2 * np.pi * (1.5 + np.arange(n)) * t[k]) + rng.normal(0, 3e-4, n)
        meas.append((0.02 * np.sin(2 * np.pi * 0.7 * t[k]), absolute_to_relative(theta)))
    x_ref = [0.0 if k < steps // 2 else 0.1 for k in range(steps)]
    ctl.reset(Measurement(0.0, meas[0][0], meas[0][1]), x_ref=x_ref[0])
    u = [ctl.step(Measurement(t[k], meas[k][0], meas[k][1]), x_ref=x_ref[k]) for k in range(steps)]
    return {
        "xRef": x_ref,
        "cartPosition": [m[0] for m in meas],
        "jointAngles": [m[1].tolist() for m in meas],
        "u": u,
    }


def plant_golden(rig: RigParams, frictionless: bool, seconds: float = 1.0) -> dict:
    """Open-loop MuJoCo response to a prescribed accel command (no sensor noise)."""
    if frictionless:
        rig = replace(
            rig,
            joints=replace(rig.joints, coulomb=0.0, viscous=0.0),
            cart=replace(rig.cart, friction=0.0, viscous=0.0),
        )
    plant = Plant(rig, seed=0)
    n = rig.n_links
    theta0 = np.deg2rad(np.linspace(0.5, -0.3, n))
    plant.reset(theta=theta0)
    steps = round(seconds * rig.timing.control_rate)
    cmd, rows = [], []
    for k in range(steps):
        u = 8.0 * np.sin(2 * np.pi * 2.0 * k * rig.control_dt)
        plant.step(u)
        s = plant.state()
        cmd.append(float(u))
        rows.append(
            {
                "x": float(s["x"]),
                "v": float(s["v"]),
                "theta": np.asarray(s["theta"]).tolist(),
                "omega": np.asarray(s["omega"]).tolist(),
                "force": float(s["force"]),
            }
        )
        if plant.fallen():
            break
    return {"rig": rig_json(rig), "theta0": theta0.tolist(), "accelCommand": cmd, "states": rows}


def main() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    index = []
    for pid, name, desc, rig in presets():
        ctl = LQRController(rig)
        doc = {
            "id": pid,
            "name": name,
            "description": desc,
            "unstablePoles": unstable_poles(rig).tolist(),
            "rig": rig_json(rig),
            "controller": controller_json(ctl),
        }
        (PRESETS / f"{pid}.json").write_text(json.dumps(doc))
        index.append({"id": pid, "name": name, "description": desc})
        golden = {"preset": pid, **controller_golden(rig, LQRController(rig))}
        (GOLDEN / f"{pid}.controller.json").write_text(json.dumps(golden))
        if pid in ("quad", "links1"):
            plant = {
                "frictionless": plant_golden(rig, frictionless=True),
                "nominal": plant_golden(rig, frictionless=False),
            }
            (GOLDEN / f"{pid}.plant.json").write_text(json.dumps(plant))
        print(f"{pid:11s} nz={ctl.nz:2d} closed-loop radius={ctl.closed_loop_radius:.6f}")
    (PRESETS / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index)} presets to {PRESETS.relative_to(ROOT)} and golden traces to {GOLDEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
