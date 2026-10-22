"""Closed loop: MuJoCo plant + sensor model + embedded-style controller.

One call to `Simulation.step()` is one control period of the microcontroller:
sample the sensors, run the controller, and hand the command to the drive after
`compute_delay_steps` periods, while the plant integrates at its own finer rate.

With a `TransitionLibrary`, `request_configuration` moves the chain between
equilibria while the loop keeps running: the cart is first brought to the
centre under the balancing controller, then each planned transition on the
route is tracked by a `TransitionController` and handed back to the balancing
controller of the configuration it reaches.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from quad_pendulum.controller import LQRController
from quad_pendulum.params import RigParams, equilibrium_angles, parse_configuration, wrap_angle
from quad_pendulum.plant import Plant
from quad_pendulum.transition import TransitionConfig, TransitionController, TransitionLibrary

# A transition starts once the cart is this close to the centre and the chain is
# this still, held for SETTLE_TIME (estimated state, as the firmware would see it).
CENTER_TOL = 0.05  # m (the planned motion is shifted to start where the cart is)
CENTER_SPEED_TOL = 0.03  # m/s
SETTLE_ANGLE_TOL = np.deg2rad(2.0)  # rad
SETTLE_RATE_TOL = 0.2  # rad/s
SETTLE_TIME = 0.2  # s


class Simulation:
    def __init__(
        self,
        rig: RigParams | None = None,
        *,
        controller_rig: RigParams | None = None,
        seed: int | None = 0,
        configuration: str | None = None,
        transitions: TransitionLibrary | None = None,
        transition_config: TransitionConfig | None = None,
    ) -> None:
        """`rig` is the true plant; `controller_rig` is what the controller believes
        (defaults to the same). Pass a perturbed `rig` to test model mismatch.
        `configuration` picks the equilibrium to balance ("UUUD" = top link down,
        bottom link first); the default is all upright. `transitions` enables
        `request_configuration`."""
        self.rig = rig or RigParams()
        self.controller_rig = controller_rig or self.rig
        self.configuration = parse_configuration(configuration, self.rig.n_links)
        self.theta_eq = equilibrium_angles(self.configuration, self.rig.n_links)
        self.plant = Plant(self.rig, seed=seed)
        self.transitions = transitions
        self.transition_config = transition_config
        self._controllers: dict[str, LQRController] = {}
        self._trackers: dict[tuple[str, str], TransitionController] = {}
        self.controller = self._balancer(self.configuration)
        self.mode = "hold"  # hold | center | transition
        self.target = self.configuration
        self.route: list[tuple[str, str]] = []
        self.tracker: TransitionController | None = None
        self._settled = 0
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
        """Start at the target equilibrium (or at `theta`, absolute radians),
        optionally with a random tilt of each link about the equilibrium. The controller is told the true starting pose, like a
        rig released from a holding jig after a sensor read."""
        n = self.rig.n_links
        if theta is None:
            theta = self.theta_eq + np.deg2rad(self.rng.uniform(-random_tilt_deg, random_tilt_deg, n))
        self.plant.reset(theta=np.asarray(theta, dtype=float), cart_x=cart_x)
        self.controller.reset(self.plant.measure())
        self.mode, self.target, self.route, self.tracker = "hold", self.configuration, [], None
        self._pending = deque([0.0] * self.rig.timing.compute_delay_steps)
        self.last_command = 0.0
        self.fallen = False

    def set_reference(self, x_ref: float) -> None:
        limit = 0.8 * self.rig.cart.rail_half_length
        self.x_ref = float(np.clip(x_ref, -limit, limit))

    def push(self, link: int, force: float, duration: float = 0.02) -> None:
        self.plant.push(link, force, duration)

    def _balancer(self, configuration: str) -> LQRController:
        if configuration not in self._controllers:
            self._controllers[configuration] = LQRController(
                self.controller_rig, configuration=configuration
            )
        return self._controllers[configuration]

    def transition_controller(self, start: str, goal: str) -> TransitionController:
        """Designed once per transition and cached (the design takes a moment)."""
        key = (start, goal)
        if key not in self._trackers:
            self._trackers[key] = TransitionController(
                self.controller_rig, self.transitions.get(start, goal), self.transition_config
            )
        return self._trackers[key]

    def set_configuration(self, configuration: str) -> None:
        """Jump straight to another equilibrium (resets the plant there)."""
        self.configuration = parse_configuration(configuration, self.rig.n_links)
        self.theta_eq = equilibrium_angles(self.configuration, self.rig.n_links)
        self.controller = self._balancer(self.configuration)
        self.reset()

    def request_configuration(self, goal: str) -> bool:
        """Move to `goal` in real time along the stored transitions. Returns False
        if there is no library or no route; ignored while a transition runs."""
        goal = parse_configuration(goal, self.rig.n_links)
        if self.fallen or self.transitions is None or self.mode == "transition":
            return False
        route = self.transitions.route(self.configuration, goal)
        if route is None:
            return False
        # Transitions are planned with the cart centred; it stays there afterwards.
        self.x_ref = 0.0
        self.target, self.route = goal, route
        self.mode = "center" if route else "hold"
        self._settled = 0
        return True

    def _ready_to_start(self) -> bool:
        z = self.controller.physical_estimate
        n = self.rig.n_links
        still = (
            abs(z[0]) < CENTER_TOL
            and abs(z[1]) < CENTER_SPEED_TOL
            and np.abs(z[3 : 3 + n]).max() < SETTLE_ANGLE_TOL
            and np.abs(z[3 + n : 3 + 2 * n]).max() < SETTLE_RATE_TOL
        )
        self._settled = self._settled + 1 if still else 0
        return self._settled >= round(SETTLE_TIME / self.rig.control_dt)

    def step(self) -> None:
        meas = self.plant.measure()
        if not self.controller_enabled:
            cmd = 0.0
            self.controller.reset(meas)
            self.mode, self.route, self.tracker = "hold", [], None
        elif self.mode == "transition":
            cmd = self.tracker.step(meas)
            if self.tracker.done:
                start, goal = self.route.pop(0)
                self.configuration = goal
                self.theta_eq = equilibrium_angles(goal, self.rig.n_links)
                self.controller = self._balancer(goal)
                self.controller.seed(self.tracker.handoff(), list(self._pending)[1:] + [cmd])
                self.tracker = None
                self.mode = "center" if self.route else "hold"
                self._settled = 0
        else:
            x_ref = 0.0 if self.mode == "center" else self.x_ref
            cmd = self.controller.step(meas, x_ref=x_ref)
            if self.mode == "center" and self._ready_to_start():
                start, goal = self.route[0]
                self.tracker = self.transition_controller(start, goal)
                z = self.controller.physical_estimate
                self.tracker.begin(z, list(self._pending)[1:] + [cmd], x_offset=z[0])
                self.mode = "transition"
        self._pending.append(cmd)
        applied = self._pending.popleft()
        self.last_command = applied
        self.plant.step(applied)
        if self.mode == "transition":
            k = min(self.tracker.k, self.tracker.steps)
            nominal = self.tracker.nominal[:, k]
            n = self.rig.n_links
            theta = self.plant.state()["theta"]
            off = np.abs(wrap_angle(theta - nominal[3 : 3 + n])).max() > 0.6
            if off or self.plant.fallen(theta_eq=theta):
                self.fallen = True
        elif self.plant.fallen(theta_eq=self.theta_eq):
            self.fallen = True

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds * self.rig.timing.control_rate)):
            self.step()

    def status(self) -> dict:
        s = self.plant.state()
        return {
            "t": self.t,
            "configuration": self.configuration,
            "mode": self.mode,
            "target": self.target,
            "transition_progress": self.tracker.progress if self.tracker else None,
            "x": s["x"],
            "v": s["v"],
            "a": s["a"],
            "theta": s["theta"],
            "omega": s["omega"],
            "x_ref": self.x_ref,
            "command": self.last_command,
            "theta_estimate": (
                self.tracker.theta_estimate if self.tracker else self.controller.theta_estimate
            ),
            "fallen": self.fallen,
            "hit_rail_end": self.plant.hit_rail_end,
            "controller_enabled": self.controller_enabled,
        }
