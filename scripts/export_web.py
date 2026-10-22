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
with y = [cart_position - x_ref, wrap(cumsum(joint_angles) - theta_eq)].
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from quad_pendulum.controller import LQRController
from quad_pendulum.linear import unstable_poles
from quad_pendulum.params import (
    Measurement,
    RigParams,
    absolute_to_relative,
    all_configurations,
    equilibrium_angles,
    wrap_angle,
)
from quad_pendulum.plant import Plant
from quad_pendulum.presets import presets
from quad_pendulum.transition import DATA_DIR, Trajectory, TransitionConfig, TransitionController

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "web" / "public" / "presets"
GOLDEN = ROOT / "web" / "test" / "golden"


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
        "configuration": ctl.configuration,
        "thetaEq": ctl.theta_eq.tolist(),
        "nz": int(ctl.nz),
        "nx": int(ctl.nx),
        "delay": int(ctl.delay),
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
        theta = ctl.theta_eq + 0.004 * np.sin(2 * np.pi * (1.5 + np.arange(n)) * t[k]) + rng.normal(0, 3e-4, n)
        q = absolute_to_relative(theta)
        if "D" in ctl.configuration:
            # Encoders cross +/-pi at folded joints; whole turns must not change y.
            q = wrap_angle(q)
            if k % 7 == 0:
                q[k % n] += 2 * np.pi * (-1 if k % 2 else 1)
        meas.append((0.02 * np.sin(2 * np.pi * 0.7 * t[k]), q))
    x_ref = [0.0 if k < steps // 2 else 0.1 for k in range(steps)]
    ctl.reset(Measurement(0.0, meas[0][0], meas[0][1]), x_ref=x_ref[0])
    u = [ctl.step(Measurement(t[k], meas[k][0], meas[k][1]), x_ref=x_ref[k]) for k in range(steps)]
    return {
        "configuration": ctl.configuration,
        "xRef": x_ref,
        "cartPosition": [m[0] for m in meas],
        "jointAngles": [m[1].tolist() for m in meas],
        "u": u,
    }


def plant_golden(rig: RigParams, frictionless: bool, seconds: float = 1.0, configuration: str | None = None) -> dict:
    """Open-loop MuJoCo response to a prescribed accel command (no sensor noise)."""
    if frictionless:
        rig = replace(
            rig,
            joints=replace(rig.joints, coulomb=0.0, viscous=0.0),
            cart=replace(rig.cart, friction=0.0, viscous=0.0),
        )
    plant = Plant(rig, seed=0)
    n = rig.n_links
    theta_eq = equilibrium_angles(configuration, n)
    theta0 = theta_eq + np.deg2rad(np.linspace(0.5, -0.3, n))
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
        if plant.fallen(theta_eq=theta_eq):
            break
    return {"rig": rig_json(rig), "theta0": theta0.tolist(), "accelCommand": cmd, "states": rows}


def export_transitions(pid: str) -> list[dict]:
    """Snapshot completed files without modifying the library being generated."""
    source = DATA_DIR / pid
    target = PRESETS / pid / "transitions"
    target.mkdir(parents=True, exist_ok=True)
    index_path = source / "index.json"
    if index_path.exists():
        entries = json.loads(index_path.read_text())
    else:
        entries = []
        for path in sorted(source.glob("*-*.json")):
            # Generators publish completed trajectories atomically. If an older
            # generator is still writing one, leave it for the next export.
            try:
                doc = json.loads(path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                continue
            entries.append({k: doc[k] for k in ("start", "goal", "duration")} | {"file": path.name})
    for entry in entries:
        shutil.copyfile(source / entry["file"], target / entry["file"])
    (target / "index.json").write_text(json.dumps(entries, indent=2))
    return entries


def transition_golden(rig: RigParams) -> dict:
    trajectory = Trajectory.load(DATA_DIR / "links2" / "DD-UU.json")
    ctl = TransitionController(rig, trajectory)
    deviation = np.array([0.017, -0.006, 0.025, 0.001, -0.0015, 0.004, -0.003])
    in_flight = np.linspace(0.12, 0.18, ctl.delay)
    offset = 0.017
    ctl.begin(deviation, in_flight, offset)
    checkpoints = sorted({0, 1, 37, ctl.steps // 2, ctl.steps - 1})
    measurements, commands, states = [], [], {}
    for k in range(ctl.steps):
        nominal = ctl.nominal[:, k]
        theta = nominal[3:3 + ctl.n] + 0.0002 * np.sin(0.13 * k + np.arange(ctl.n))
        cart = nominal[0] + offset + 0.00002 * np.cos(0.17 * k)
        joints = wrap_angle(absolute_to_relative(theta))
        meas = Measurement(k * ctl.dt, cart, joints)
        measurements.append({"cartPosition": float(cart), "jointAngles": joints.tolist()})
        commands.append(ctl.step(meas))
        if k in checkpoints:
            states[str(k)] = ctl.z.tolist()
    return {
        "preset": "links2", "deviation": deviation.tolist(), "inFlight": in_flight.tolist(),
        "xOffset": offset, "measurements": measurements, "u": commands, "z": states,
        "gains": {str(k): ctl.gains[k].tolist() for k in checkpoints},
        "nominal": {str(k): ctl.nominal[:, k].tolist() for k in checkpoints},
        "az": {str(k): ctl.az[k].tolist() for k in checkpoints},
        "bz": {str(k): ctl.bz[k].tolist() for k in checkpoints},
    }


def main() -> None:
    PRESETS.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    index = []
    for pid, name, desc, rig in presets():
        configurations = {}
        for cfg in all_configurations(rig.n_links):
            ctl = LQRController(rig, configuration=cfg)
            configurations[cfg] = {
                "controller": controller_json(ctl),
                "unstablePoles": unstable_poles(rig, cfg).tolist(),
            }
        default = configurations["U" * rig.n_links]
        doc = {
            "id": pid,
            "name": name,
            "description": desc,
            "unstablePoles": default["unstablePoles"],
            "rig": rig_json(rig),
            "controller": default["controller"],
            "configurations": configurations,
            "transitions": export_transitions(pid),
            "transitionConfig": camel(asdict(TransitionConfig())),
        }
        (PRESETS / f"{pid}.json").write_text(json.dumps(doc, separators=(",", ":")))
        index.append({"id": pid, "name": name, "description": desc})
        golden = {"preset": pid, **controller_golden(rig, LQRController(rig))}
        (GOLDEN / f"{pid}.controller.json").write_text(json.dumps(golden))
        if pid == "quad":
            for cfg in ("UUUD", "DDUU"):
                golden = {"preset": pid, **controller_golden(rig, LQRController(rig, configuration=cfg))}
                (GOLDEN / f"{pid}.{cfg}.controller.json").write_text(json.dumps(golden))
            mixed = {
                mode: plant_golden(
                    rig, frictionless=mode == "frictionless", seconds=0.25, configuration="UUUD"
                )
                for mode in ("frictionless", "nominal")
            }
            (GOLDEN / f"{pid}.UUUD.plant.json").write_text(json.dumps(mixed))
        if pid in ("quad", "links1"):
            plant = {
                "frictionless": plant_golden(rig, frictionless=True),
                "nominal": plant_golden(rig, frictionless=False),
            }
            (GOLDEN / f"{pid}.plant.json").write_text(json.dumps(plant))
        if pid == "links2":
            (GOLDEN / "links2.transition.json").write_text(json.dumps(transition_golden(rig)))
        print(f"{pid:11s} configurations={len(configurations):2d} nz={default['controller']['nz']:2d}")
    (PRESETS / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index)} presets to {PRESETS.relative_to(ROOT)} and golden traces to {GOLDEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
