#!/usr/bin/env python
"""Train an RL policy for the cart-pendulum environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from quad_pendulum import QuadPendulumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_env(links: int, seed: int):
    def _factory():
        env = QuadPendulumEnv(links=links)
        env.reset(seed=seed)
        return env

    return _factory


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.check_env:
        check_env(QuadPendulumEnv(links=args.links), warn=True)

    train_env = VecMonitor(DummyVecEnv([make_env(args.links, args.seed)]))
    eval_env = VecMonitor(DummyVecEnv([make_env(args.links, args.seed + 10_000)]))

    run_name = f"{args.algo}_links{args.links}"
    tensorboard_log = args.log_dir / run_name

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            learning_starts=10_000,
            ent_coef="auto",
            verbose=1,
            tensorboard_log=str(tensorboard_log),
            seed=args.seed,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=str(tensorboard_log),
            seed=args.seed,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(args.model_dir),
        name_prefix=run_name,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.model_dir / f"{run_name}_best"),
        log_path=str(args.log_dir / f"{run_name}_eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=run_name,
    )
    model_path = args.model_dir / f"{run_name}.zip"
    if model_path.exists() and not args.overwrite:
        model_path = _next_available_model_path(args.model_dir, run_name)
    model.save(model_path)


def _next_available_model_path(model_dir: Path, run_name: str) -> Path:
    for index in range(2, 10_000):
        candidate = model_dir / f"{run_name}_run{index}.zip"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not find an available model filename")


if __name__ == "__main__":
    main()
