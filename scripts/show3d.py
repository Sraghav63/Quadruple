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

from quad_pendulum.params import RigParams
from quad_pendulum.sim import Simulation

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
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def generate_trajectory(seconds: float, seed: int) -> dict:
    rig = RigParams()
    sim = Simulation(rig, seed=seed)
    sim.reset(random_tilt_deg=0.05)
    script = sorted(SCRIPT, key=lambda item: item[0])
    frames = []
    for frame in range(int(seconds * FPS)):
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
        "linkLengths": [link.length for link in rig.links],
        "railHalfLength": rig.cart.rail_half_length,
        "fps": FPS,
        "control": "lqr",
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
    trajectory = generate_trajectory(args.seconds, args.seed)
    TRAJECTORY_PATH.write_text(json.dumps(trajectory), encoding="utf-8")
    print(f"Wrote {TRAJECTORY_PATH} ({len(trajectory['frames'])} frames)")
    if trajectory["fallen"]:
        print("Warning: the pendulum fell during the run.")

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
