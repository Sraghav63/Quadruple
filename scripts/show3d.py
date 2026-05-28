#!/usr/bin/env python
"""Generate a rollout and serve a browser-based 3D viewer."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
from pathlib import Path
import socketserver
import webbrowser

import numpy as np
from stable_baselines3 import PPO, SAC

from quad_pendulum import QuadPendulumEnv
from quad_pendulum.policy import default_model_path

ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = ROOT / "viewer3d"
TRAJECTORY_PATH = VIEWER_DIR / "trajectory.json"


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reset-noise", type=float, default=0.04)
    parser.add_argument("--control", choices=("policy", "zero"), default="policy")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    trajectory = generate_trajectory(args)
    TRAJECTORY_PATH.write_text(json.dumps(trajectory), encoding="utf-8")
    print(f"Wrote {TRAJECTORY_PATH}")

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


def bind_server(handler, preferred_port: int) -> ReusableTCPServer:
    last_error = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            return ReusableTCPServer(("127.0.0.1", port), handler)
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Could not bind a local viewer port: {last_error}")


def generate_trajectory(args: argparse.Namespace) -> dict:
    env = QuadPendulumEnv(
        links=args.links,
        reset_noise_scale=args.reset_noise,
        max_episode_steps=args.steps,
    )
    model = load_model(args) if args.control == "policy" else None

    obs, _ = env.reset(seed=0)
    frames = []
    for step in range(args.steps):
        if model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(
            {
                "cartX": float(env.data.qpos[0]),
                "cartV": float(env.data.qvel[0]),
                "angles": [float(angle) for angle in env.data.qpos[1 : 1 + env.links]],
                "action": float(np.asarray(action).reshape(-1)[0]),
                "reset": bool(terminated or truncated),
            }
        )

        if terminated or truncated:
            obs, _ = env.reset(seed=step + 1)

    env.close()
    return {
        "links": args.links,
        "linkLength": 0.45,
        "fps": 50,
        "control": args.control,
        "frames": frames,
    }


def load_model(args: argparse.Namespace):
    model_path = default_model_path(args.algo, args.links, args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Use --model or --control zero.")
    model_cls = SAC if args.algo == "sac" else PPO
    return model_cls.load(model_path)


if __name__ == "__main__":
    main()
