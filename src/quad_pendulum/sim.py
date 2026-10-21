"""Closed loop: MuJoCo plant + sensor model + embedded-style controller.

One call to `Simulation.step()` is one control period of the microcontroller:
sample the sensors, run the controller, and hand the command to the drive after
`compute_delay_steps` periods, while the plant integrates at its own finer rate.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from quad_pendulum.controller import LQRController
from quad_pendulum.params import RigParams
from quad_pendulum.plant import Plant


class Simulation:
    def __init__(
        self,
        rig: RigParams | None = None,
        *,
        controller_rig: RigParams | None = None,
        seed: int | None = 0,
    ) -> None:
        """`rig` is the true plant; `controller_rig` is what the controller believes
        (defaults to the same). Pass a perturbed `rig` to test model mismatch."""
        self.rig = rig or RigParams()
        self.plant = Plant(self.rig, seed=seed)
        self.controller = LQRController(controller_rig or self.rig)
        self.rng = np.random.default_rng(seed)
        self.controller_enabled = True
        self.x_ref = 0.0
        self.last_command = 0.0
        self._pending: deque[float] = deque()
        self.fallen = False
        self.reset()

    @property
    def t(self) -> float:
        return self.plant.t

    def reset(self, theta=None, *, random_tilt_deg: float = 0.0, cart_x: float = 0.0) -> None:
        """Start upright (or at `theta`, absolute radians), optionally with random
        tilt of each link. The controller is told the true starting pose, like a
        rig released from a holding jig after a sensor read."""
        n = self.rig.n_links
        if theta is None:
            theta = np.deg2rad(self.rng.uniform(-random_tilt_deg, random_tilt_deg, n))
        self.plant.reset(theta=np.asarray(theta, dtype=float), cart_x=cart_x)
        self.controller.reset(self.plant.measure())
        self._pending = deque([0.0] * self.rig.timing.compute_delay_steps)
        self.last_command = 0.0
        self.fallen = False

    def set_reference(self, x_ref: float) -> None:
        limit = 0.8 * self.rig.cart.rail_half_length
        self.x_ref = float(np.clip(x_ref, -limit, limit))

    def push(self, link: int, force: float, duration: float = 0.02) -> None:
        self.plant.push(link, force, duration)

    def step(self) -> None:
        meas = self.plant.measure()
        if self.controller_enabled:
            cmd = self.controller.step(meas, x_ref=self.x_ref)
        else:
            cmd = 0.0
            self.controller.reset(meas)
        self._pending.append(cmd)
        applied = self._pending.popleft()
        self.last_command = applied
        self.plant.step(applied)
        if self.plant.fallen():
            self.fallen = True

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds * self.rig.timing.control_rate)):
            self.step()

    def status(self) -> dict:
        s = self.plant.state()
        return {
            "t": self.t,
            "x": s["x"],
            "v": s["v"],
            "a": s["a"],
            "theta": s["theta"],
            "omega": s["omega"],
            "x_ref": self.x_ref,
            "command": self.last_command,
            "theta_estimate": self.controller.theta_estimate,
            "fallen": self.fallen,
            "hit_rail_end": self.plant.hit_rail_end,
            "controller_enabled": self.controller_enabled,
        }
