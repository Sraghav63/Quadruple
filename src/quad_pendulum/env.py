"""Gymnasium environment for RL balancing of a cart-pendulum."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from quad_pendulum.model import make_cart_pendulum_xml


@dataclass(frozen=True)
class RewardWeights:
    angle: float = 2.0
    angular_velocity: float = 0.04
    cart_position: float = 0.2
    cart_velocity: float = 0.02
    control: float = 0.001
    alive: float = 1.0


class QuadPendulumEnv(gym.Env):
    """Balance a 1- to 4-link inverted pendulum on a sliding cart.

    The action is one normalized horizontal cart force in ``[-1, 1]``. The
    observation is cart position, cart velocity, then ``sin(theta)``,
    ``cos(theta)``, and angular velocity for each link.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        links: int = 4,
        *,
        render_mode: str | None = None,
        frame_skip: int = 5,
        force_scale: float = 80.0,
        reset_noise_scale: float = 0.04,
        max_cart_position: float = 2.4,
        max_episode_steps: int = 1000,
        terminate_cos_threshold: float = -0.3,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.links = links
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.force_scale = force_scale
        self.reset_noise_scale = reset_noise_scale
        self.max_cart_position = max_cart_position
        self.max_episode_steps = max_episode_steps
        self.terminate_cos_threshold = terminate_cos_threshold
        self.reward_weights = reward_weights or RewardWeights()
        self.elapsed_steps = 0

        xml = make_cart_pendulum_xml(links=links)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._camera_configured = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_size = 2 + (3 * links)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.elapsed_steps = 0

        noise_scale = self.reset_noise_scale
        if options and "noise_scale" in options:
            noise_scale = float(options["noise_scale"])

        self.data.qpos[:] = self.np_random.uniform(-noise_scale, noise_scale, self.model.nq)
        self.data.qpos[0] = self.np_random.uniform(-0.05, 0.05)
        self.data.qvel[:] = self.np_random.uniform(-noise_scale, noise_scale, self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[0] = float(clipped_action[0]) * self.force_scale

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.elapsed_steps += 1
        obs = self._get_obs()
        reward, reward_info = self._reward(float(clipped_action[0]))
        terminated = self._is_terminated()
        truncated = self.elapsed_steps >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, reward_info

    def render(self):
        if self.render_mode is None:
            return None

        if self.viewer is None:
            if self.render_mode == "human":
                mujoco_viewer = importlib.import_module("mujoco.viewer")

                self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
                self._configure_human_camera()
            else:
                self.viewer = mujoco.Renderer(self.model, width=1280, height=720)

        if self.render_mode == "human":
            self.viewer.sync()
            return None

        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
        self.viewer.update_scene(self.data, camera=camera_id)
        return self.viewer.render()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            self._camera_configured = False

    def viewer_is_running(self) -> bool:
        if self.viewer is None:
            return True
        is_running = getattr(self.viewer, "is_running", None)
        return True if is_running is None else bool(is_running())

    def _configure_human_camera(self) -> None:
        if self._camera_configured or self.viewer is None:
            return

        self.viewer.cam.lookat[:] = [0.0, 0.0, 0.72]
        self.viewer.cam.distance = 4.2
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -14
        self._camera_configured = True

    def _get_obs(self) -> np.ndarray:
        cart_position = self.data.qpos[0]
        cart_velocity = self.data.qvel[0]
        angles = self.data.qpos[1 : 1 + self.links]
        angular_velocities = self.data.qvel[1 : 1 + self.links]

        obs = [cart_position, cart_velocity]
        for angle, angular_velocity in zip(angles, angular_velocities, strict=True):
            obs.extend((np.sin(angle), np.cos(angle), angular_velocity))

        return np.asarray(obs, dtype=np.float32)

    def _reward(self, action: float) -> tuple[float, dict]:
        weights = self.reward_weights
        cart_position = float(self.data.qpos[0])
        cart_velocity = float(self.data.qvel[0])
        angles = self.data.qpos[1 : 1 + self.links]
        angular_velocities = self.data.qvel[1 : 1 + self.links]

        angle_cost = float(np.sum(1.0 - np.cos(angles)))
        angular_velocity_cost = float(np.sum(np.square(angular_velocities)))
        cart_position_cost = cart_position * cart_position
        cart_velocity_cost = cart_velocity * cart_velocity
        control_cost = action * action

        reward = (
            weights.alive
            - weights.angle * angle_cost
            - weights.angular_velocity * angular_velocity_cost
            - weights.cart_position * cart_position_cost
            - weights.cart_velocity * cart_velocity_cost
            - weights.control * control_cost
        )

        return float(reward), {
            "angle_cost": angle_cost,
            "angular_velocity_cost": angular_velocity_cost,
            "cart_position_cost": cart_position_cost,
            "cart_velocity_cost": cart_velocity_cost,
            "control_cost": control_cost,
        }

    def _is_terminated(self) -> bool:
        if abs(float(self.data.qpos[0])) > self.max_cart_position:
            return True

        link_cosines = np.cos(self.data.qpos[1 : 1 + self.links])
        return bool(np.any(link_cosines < self.terminate_cos_threshold))
