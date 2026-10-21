"""Ground-truth mechanics, velocity-mode servo, disturbances, and sensors."""

from __future__ import annotations

import mujoco
import numpy as np

from quad_pendulum.model import make_rig_xml
from quad_pendulum.params import (
    Measurement,
    RigParams,
    absolute_to_relative,
    relative_to_absolute,
)


class Plant:
    def __init__(self, rig: RigParams, seed: int | None = None) -> None:
        self.rig = rig
        self._substeps = rig.physics_substeps
        self.model = mujoco.MjModel.from_xml_string(make_rig_xml(rig))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(seed)
        self._link_ids = np.array([self.model.body(f"link{i + 1}").id for i in range(rig.n_links)])
        self._tip_id = self.model.site(f"tip{rig.n_links}").id
        mass_estimate = rig.cart.mass + sum(link.mass for link in rig.links)
        self._kp = mass_estimate * rig.drive.velocity_bandwidth
        self._ki = self._kp * rig.drive.velocity_bandwidth / 4
        self.reset()

    @property
    def t(self) -> float:
        return float(self.data.time)

    def reset(self, theta: np.ndarray | None = None, cart_x: float = 0.0) -> None:
        """Release at absolute link angles, with a stationary, reset drive."""
        angles = np.zeros(self.rig.n_links) if theta is None else np.asarray(theta, dtype=float)
        if angles.shape != (self.rig.n_links,) or not np.all(np.isfinite(angles)):
            raise ValueError("theta must contain one finite absolute angle per link")
        if not np.isfinite(cart_x):
            raise ValueError("cart_x must be finite")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = cart_x
        self.data.qpos[1:] = absolute_to_relative(angles)
        self._v_sp = 0.0
        self._integral_error = 0.0
        self._force = 0.0
        self._pushes: list[tuple[int, float, float]] = []
        self.hit_rail_end = abs(cart_x) > self.rig.cart.rail_half_length
        mujoco.mj_forward(self.model, self.data)

    def step(self, accel_cmd: float) -> None:
        """Advance one control period, closing the servo at the physics rate."""
        if not np.isfinite(accel_cmd):
            raise ValueError("accel_cmd must be finite")
        drive = self.rig.drive
        accel = float(np.clip(accel_cmd, -drive.accel_limit, drive.accel_limit))
        dt = self.model.opt.timestep
        for _ in range(self._substeps):
            self._v_sp = float(
                np.clip(self._v_sp + accel * dt, -drive.velocity_limit, drive.velocity_limit)
            )
            error = self._v_sp - float(self.data.qvel[0])
            candidate = self._integral_error + error * dt
            candidate_force = self._kp * error + self._ki * candidate
            # Conditional integration: retain corrections that unwind saturation.
            if abs(candidate_force) <= drive.force_limit or candidate_force * error <= 0:
                self._integral_error = candidate
            self._force = float(
                np.clip(
                    self._kp * error + self._ki * self._integral_error,
                    -drive.force_limit,
                    drive.force_limit,
                )
            )
            self.data.ctrl[0] = self._force
            self._apply_pushes(dt)
            mujoco.mj_step(self.model, self.data)
            if abs(self.data.qpos[0]) > self.rig.cart.rail_half_length:
                self.hit_rail_end = True
            # mj_step leaves position-derived fields at the start of the step.
            # Refresh tip/CoM locations and report acceleration at the new state.
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self._apply_pushes()
            mujoco.mj_forward(self.model, self.data)

    def measure(self) -> Measurement:
        """Sample only cart position and relative joint encoder angles."""
        sensors = self.rig.sensors
        cart = self.data.qpos[0] + self._rng.normal(0.0, sensors.cart_noise)
        q = self.data.qpos[1:] + self._rng.normal(0.0, sensors.encoder_noise, self.rig.n_links)
        quantum = 2 * np.pi / 2**sensors.encoder_bits
        return Measurement(
            t=self.t,
            cart_position=float(np.rint(cart / sensors.cart_resolution) * sensors.cart_resolution),
            joint_angles=np.rint(q / quantum) * quantum,
        )

    def push(self, link: int, force: float, duration: float = 0.02) -> None:
        """Add a world +x force at a link's moving top; overlapping pushes add."""
        if not isinstance(link, (int, np.integer)) or not 0 <= link < self.rig.n_links:
            raise IndexError("link must be a zero-based link index")
        if not np.isfinite(force) or not np.isfinite(duration) or duration <= 0:
            raise ValueError("force must be finite and duration must be positive and finite")
        self._pushes.append((int(link), float(force), self.t + float(duration)))
        self._apply_pushes()
        mujoco.mj_forward(self.model, self.data)

    def _apply_pushes(self, dt: float = 0.0) -> None:
        self.data.xfrc_applied[:] = 0.0
        self._pushes = [push for push in self._pushes if push[2] > self.t + 1e-12]
        for link, force, end in self._pushes:
            fraction = min(1.0, (end - self.t) / dt) if dt else 1.0
            body_id = self._link_ids[link]
            top = (
                self.data.xpos[body_id]
                + self.data.xmat[body_id].reshape(3, 3)[:, 2] * self.rig.links[link].length
            )
            applied = np.array([force * fraction, 0.0, 0.0])
            self.data.xfrc_applied[body_id, :3] += applied
            self.data.xfrc_applied[body_id, 3:] += np.cross(top - self.data.xipos[body_id], applied)

    def state(self) -> dict[str, float | np.ndarray]:
        """Return ground truth for drawing and diagnostics, never sensor feedback."""
        q = self.data.qpos[1:].copy()
        return {
            "x": float(self.data.qpos[0]),
            "v": float(self.data.qvel[0]),
            "a": float(self.data.qacc[0]),
            "theta": relative_to_absolute(q),
            "omega": relative_to_absolute(self.data.qvel[1:]),
            "q": q,
            "force": self._force,
            "v_setpoint": self._v_sp,
        }

    def fallen(self, limit_rad: float = 0.6) -> bool:
        if abs(self.data.qpos[0]) > self.rig.cart.rail_half_length:
            self.hit_rail_end = True
        return bool(
            self.hit_rail_end
            or np.any(np.abs(relative_to_absolute(self.data.qpos[1:])) > limit_rad)
        )

    def link_points(self) -> np.ndarray:
        """World (x, z) of every lower hinge followed by the chain tip."""
        return np.vstack(
            (self.data.xpos[self._link_ids][:, [0, 2]], self.data.site_xpos[self._tip_id, [0, 2]])
        )
