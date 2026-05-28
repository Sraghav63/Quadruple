#!/usr/bin/env python
"""Run the cart-pendulum with a stable pygame viewer.

The default viewer deliberately avoids MuJoCo's interactive GLFW window because
that path is unstable on some Wayland/libdecor setups.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from stable_baselines3 import PPO, SAC

from quad_pendulum import QuadPendulumEnv
from quad_pendulum.policy import default_model_path

LINK_LENGTH = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument("--reset-noise", type=float, default=0.04)
    parser.add_argument("--control", choices=("policy", "mouse", "zero"), default="policy")
    parser.add_argument("--mujoco-viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mujoco_viewer:
        run_mujoco_viewer(args)
    else:
        run_pygame_viewer(args)


def run_pygame_viewer(args: argparse.Namespace) -> None:
    env = QuadPendulumEnv(links=args.links, reset_noise_scale=args.reset_noise)
    model = load_model(args) if args.control == "policy" else None

    pygame.init()
    screen = pygame.display.set_mode((1100, 700), pygame.RESIZABLE)
    pygame.display.set_caption("Quad Pendulum RL")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 18)

    obs, _ = env.reset(seed=0)
    target_x = float(env.data.qpos[0])
    episode = 0
    running = True

    while running:
        dt = env.model.opt.timestep * env.frame_skip * max(args.slowdown, 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset(seed=episode)
                    target_x = float(env.data.qpos[0])
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                if args.control == "mouse" and (event.type == pygame.MOUSEBUTTONDOWN or event.buttons[0]):
                    target_x = screen_to_world_x(event.pos[0], screen.get_width())

        if args.control == "policy":
            action, _ = model.predict(obs, deterministic=True)
        elif args.control == "mouse":
            action = mouse_action(env, target_x)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            episode += 1
            if episode >= args.episodes and args.control != "mouse":
                break
            obs, _ = env.reset(seed=episode)
            target_x = float(env.data.qpos[0])
        elif env.elapsed_steps >= args.steps:
            episode += 1
            if episode >= args.episodes and args.control != "mouse":
                break
            obs, _ = env.reset(seed=episode)
            target_x = float(env.data.qpos[0])

        draw_scene(screen, font, env, args.control, target_x)
        pygame.display.flip()
        clock.tick(max(1, int(1.0 / dt)))

    env.close()
    pygame.quit()


def load_model(args: argparse.Namespace):
    model_path = default_model_path(args.algo, args.links, args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Use --model or --control mouse.")
    model_cls = SAC if args.algo == "sac" else PPO
    return model_cls.load(model_path)


def mouse_action(env: QuadPendulumEnv, target_x: float) -> np.ndarray:
    cart_x = float(env.data.qpos[0])
    cart_v = float(env.data.qvel[0])
    error = target_x - cart_x
    force = 5.0 * error - 0.9 * cart_v
    return np.asarray([np.clip(force, -1.0, 1.0)], dtype=np.float32)


def draw_scene(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: QuadPendulumEnv,
    control: str,
    target_x: float,
) -> None:
    width, height = screen.get_size()
    screen.fill((236, 239, 240))

    origin_y = int(height * 0.68)
    rail_y = origin_y
    scale = min(width / 7.2, height / 3.1)
    center_x = width // 2

    def to_screen(x: float, z: float = 0.0) -> tuple[int, int]:
        return int(center_x + x * scale), int(rail_y - z * scale)

    rail_left = to_screen(-3.0, 0)[0]
    rail_right = to_screen(3.0, 0)[0]
    pygame.draw.line(screen, (35, 38, 40), (rail_left, rail_y), (rail_right, rail_y), 6)
    pygame.draw.line(screen, (110, 116, 118), (rail_left, rail_y + 18), (rail_right, rail_y + 18), 3)

    for marker_x, color in [(0.0, (45, 48, 50)), (-2.4, (168, 40, 32)), (2.4, (168, 40, 32))]:
        x, _ = to_screen(marker_x, 0)
        pygame.draw.line(screen, color, (x, rail_y - 24), (x, rail_y + 24), 2)

    if control == "mouse":
        tx, _ = to_screen(target_x, 0)
        pygame.draw.line(screen, (25, 120, 190), (tx, rail_y - 46), (tx, rail_y + 46), 2)
        pygame.draw.circle(screen, (25, 120, 190), (tx, rail_y - 50), 7)

    cart_x = float(env.data.qpos[0])
    cart_screen_x, cart_screen_y = to_screen(cart_x, 0.08)
    cart_w = max(64, int(0.42 * scale))
    cart_h = max(38, int(0.19 * scale))
    cart_rect = pygame.Rect(0, 0, cart_w, cart_h)
    cart_rect.center = (cart_screen_x, cart_screen_y)
    pygame.draw.rect(screen, (20, 72, 132), cart_rect, border_radius=6)
    pygame.draw.rect(screen, (232, 238, 242), cart_rect.inflate(-22, -18), border_radius=4)

    wheel_y = cart_rect.bottom + 9
    for wheel_x in (cart_rect.left + cart_w * 0.25, cart_rect.right - cart_w * 0.25):
        pygame.draw.circle(screen, (24, 25, 28), (int(wheel_x), wheel_y), 13)
        pygame.draw.circle(screen, (170, 174, 178), (int(wheel_x), wheel_y), 5)

    pivot = to_screen(cart_x, 0.18)
    pygame.draw.circle(screen, (160, 166, 170), pivot, 12)

    cumulative_angle = 0.0
    current = np.asarray([cart_x, 0.18], dtype=np.float64)
    colors = [(205, 67, 48), (35, 138, 85), (230, 169, 42), (110, 76, 165)]
    angles = env.data.qpos[1 : 1 + env.links]
    for index, angle in enumerate(angles):
        cumulative_angle += float(angle)
        next_point = current + np.asarray(
            [np.sin(cumulative_angle) * LINK_LENGTH, np.cos(cumulative_angle) * LINK_LENGTH]
        )
        start = to_screen(float(current[0]), float(current[1]))
        end = to_screen(float(next_point[0]), float(next_point[1]))
        pygame.draw.line(screen, colors[index], start, end, 10)
        pygame.draw.circle(screen, (235, 238, 240), start, 8)
        pygame.draw.circle(screen, colors[index], end, 13)
        current = next_point

    text = (
        f"mode: {control}    cart x: {cart_x:+.2f}    "
        "drag mouse to move cart target, R reset, Esc quit"
    )
    label = font.render(text, True, (28, 32, 34))
    screen.blit(label, (18, 18))


def screen_to_world_x(mouse_x: int, width: int) -> float:
    scale = width / 7.2
    return float(np.clip((mouse_x - width / 2) / scale, -2.4, 2.4))


def run_mujoco_viewer(args: argparse.Namespace) -> None:
    env = QuadPendulumEnv(
        links=args.links,
        render_mode="human",
        reset_noise_scale=args.reset_noise,
    )
    model = load_model(args) if args.control == "policy" else None

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        env.render()
        while not done and env.viewer_is_running():
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if args.slowdown > 0:
                time.sleep(env.model.opt.timestep * env.frame_skip * args.slowdown)

    env.close()


if __name__ == "__main__":
    main()
