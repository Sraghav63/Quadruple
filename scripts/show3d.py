#!/usr/bin/env python
"""Simulate a scripted run headless and serve a browser-based 3D viewer."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import webbrowser
from pathlib import Path

from quad_pendulum.params import parse_configuration
from quad_pendulum.presets import preset_rig
from quad_pendulum.sim import Simulation
from quad_pendulum.transition import DATA_DIR, Trajectory, TransitionLibrary

ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = ROOT / "viewer3d"
TRAJECTORY_PATH = VIEWER_DIR / "trajectory.json"
FPS = 60
PUSH_FORCE = 0.1

# (time s, action, argument): push link N with force, or move the cart reference.
SCRIPT = [
    (1.0, "ref", 0.3),
    (2.0, "push", (4, PUSH_FORCE)),
    (4.0, "ref", -0.3),
    (5.0, "push", (2, -PUSH_FORCE)),
    (7.0, "ref", 0.0),
]


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds", type=float, default=None, help="simulated time (default 10, or 30 with --goto)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--config",
        default=None,
        metavar="UUUD",
        help="starting configuration: links balanced up (U) or down (D), bottom link first; "
        "default all up (all down with --goto)",
    )
    parser.add_argument(
        "--goto",
        default=None,
        metavar="CFG",
        help="request this configuration at t = 0.5 s and show the planned swing; the "
        "tap/reference script is skipped",
    )
    parser.add_argument("--preset", default="quad", help="rig preset (default quad)")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    try:
        rig = preset_rig(args.preset)
        if args.goto is not None:
            args.goto = parse_configuration(args.goto, rig.n_links)
            if args.config is None:
                args.config = "D" * rig.n_links
        args.config = parse_configuration(args.config, rig.n_links)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    if args.seconds is None:
        args.seconds = 30.0 if args.goto else 10.0
    return args


def load_library(preset: str) -> TransitionLibrary:
    directory = DATA_DIR / preset
    if (directory / "index.json").exists():
        return TransitionLibrary(directory)
    # library still being planned: use whatever edge files exist
    files = sorted(directory.glob("*-*.json")) if directory.is_dir() else []
    return TransitionLibrary(trajectories=[Trajectory.load(f) for f in files])


def generate_trajectory(
    seconds: float,
    seed: int,
    configuration: str | None = None,
    goto: str | None = None,
    preset: str = "quad",
) -> dict:
    rig = preset_rig(preset)
    library = load_library(preset) if goto else None
    sim = Simulation(rig, seed=seed, configuration=configuration, transitions=library)
    sim.reset(random_tilt_deg=0.05)
    script = [] if goto else sorted(SCRIPT, key=lambda item: item[0])
    requested = goto is None
    frames = []
    for frame in range(int(seconds * FPS)):
        if not requested and sim.t >= 0.5:
            requested = True
            if not sim.request_configuration(goto):
                raise SystemExit(f"no stored transition route {sim.configuration} -> {goto}")
        while script and script[0][0] <= sim.t:
            _, action, arg = script.pop(0)
            if action == "ref":
                sim.set_reference(arg)
            else:
                sim.push(arg[0] - 1, arg[1])
        # Step to this frame's timestamp so playback time matches sim time exactly.
        while sim.t < (frame + 1) / FPS - 1e-9:
            sim.step()
        points = sim.plant.link_points()
        frames.append(
            {
                "cartX": float(points[0][0]),
                "points": [[round(float(x), 5), round(float(z), 5)] for x, z in points],
                "ref": float(sim.x_ref),
                "reset": False,
            }
        )
    return {
        "links": rig.n_links,
        "configuration": sim.configuration,
        "linkLengths": [link.length for link in rig.links],
        "railHalfLength": rig.cart.rail_half_length,
        "fps": FPS,
        "control": "lqr" if goto is None else f"swing {configuration} to {goto}, then lqr",
        "arrived": bool(goto is None or (sim.mode == "hold" and sim.configuration == goto)),
        "fallen": bool(sim.fallen),
        "frames": frames,
    }


def bind_server(handler, preferred_port: int) -> ReusableTCPServer:
    last_error = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            return ReusableTCPServer(("127.0.0.1", port), handler)
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Could not bind a local viewer port: {last_error}")


def main() -> None:
    args = parse_args()
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = generate_trajectory(
        args.seconds, args.seed, args.config, args.goto, args.preset
    )
    TRAJECTORY_PATH.write_text(json.dumps(trajectory), encoding="utf-8")
    print(f"Wrote {TRAJECTORY_PATH} ({len(trajectory['frames'])} frames)")
    if trajectory["fallen"]:
        print("Warning: the pendulum fell during the run.")
    elif not trajectory["arrived"]:
        print("Warning: the requested configuration was not reached; try --seconds longer.")

    if args.export_only:
        return

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(VIEWER_DIR))
    server = bind_server(handler, args.port)
    with server:
        host, port = server.server_address
        url = f"http://{host}:{port}/"
        print(f"Serving 3D viewer at {url}")
        print("Press Ctrl+C to stop the server.")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
