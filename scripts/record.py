#!/usr/bin/env python
"""Record a policy rollout to an MP4 video using offscreen MuJoCo rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
from stable_baselines3 import PPO, SAC

from quad_pendulum import QuadPendulumEnv
from quad_pendulum.policy import default_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--reset-noise", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or Path("videos") / f"{args.algo}_links{args.links}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    env = QuadPendulumEnv(
        links=args.links,
        render_mode="rgb_array",
        reset_noise_scale=args.reset_noise,
        max_episode_steps=args.steps,
    )

    model = None
    if not args.zero_action:
        model_path = default_model_path(args.algo, args.links, args.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found at {model_path}. Pass --model or use --zero-action."
            )
        model_cls = SAC if args.algo == "sac" else PPO
        model = model_cls.load(model_path)

    writer = None
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            steps = 0

            while not done and steps < args.steps:
                if model is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, _, terminated, truncated, _ = env.step(action)
                frame = env.render()

                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer for {output}")

                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                done = terminated or truncated
                steps += 1
    finally:
        if writer is not None:
            writer.release()
        env.close()

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
