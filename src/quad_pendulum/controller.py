"""Discrete LQR + steady-state Kalman filter, written the way it runs on the MCU.

Everything the controller knows comes from `RigParams` (the identified model)
and the quantised `Measurement` each period. At run time it uses fixed matrix
operations, so the same gains can be exported to C (`to_c_header`)
and run on a Teensy 4.1 at 1 kHz.

Design model (see linear.py), discretised with a zero-order hold at the control
rate and augmented with the commands still in flight (compute delay):

    z = [x - x_ref, v, a, theta_1..n, omega_1..n, optional states, u in flight]

Optional leaky angle integrals, servo error and joint torque states all enter
the same exported A/B/C/K/L matrices; there is no separate runtime compensator.

Measurements: cart position and the n encoder angles (converted to absolute).

The design can be about any equilibrium with each link straight up or down
(`configuration`, e.g. "UUUD", bottom link first). The angle states are then
deviations from the equilibrium angles theta*, and the runtime subtracts
theta* (wrapped to +/-pi) from the absolute angles before correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from quad_pendulum.linear import horizontal_moments, linearized_mass, state_space
from quad_pendulum.params import (
    Measurement,
    RigParams,
    equilibrium_angles,
    parse_configuration,
    relative_to_absolute,
    wrap_angle,
)


@dataclass(frozen=True)
class ControllerConfig:
    """LQR weights (Bryson-style: 1 / acceptable_deviation^2) and filter noise levels."""

    # Trade-off on the realistic default rig: quiet estimation/effort weights
    # plus a leaky low-frequency angle cost give 2.95 m/s^2 worst-seed idle RMS,
    # 7.48 p99, all +/-0.12 N top taps, and 40/40 mismatch trials. Cart position
    # weighting also preserves the 20 cm tracking test. Bottom-link excursions
    # still reach 3.96 degrees (the 2-degree goal is missed). Higher
    # gains tighten the chain but amplify encoder noise. Rate and torque-state
    # penalties remain optional: their quieter settings lost disturbance margin.
    cart_position_tol: float = 0.08  # m
    cart_velocity_tol: float = 1.0  # m/s
    angle_tol: float = 0.2  # rad, per link
    rate_tol: float = 1.0  # rad/s
    accel_tol: float = 1.0  # m/s^2, command effort (low = less sensor-noise chatter)
    # Extra penalty on the bend at each joint (relative angles/rates). Keeps the
    # chain straight so a tap on the top link is not fixed by whipping link 1
    # far outside the small-angle region where the linear model holds.
    joint_angle_tol: float | None = 0.02  # rad
    joint_rate_tol: float | None = None  # rad/s (raises chatter; off by default)
    joint_angle_weights: tuple[float, ...] | None = None  # dimensionless per-joint cost multipliers
    # Kalman process noise: unmodelled cart acceleration and link angular kicks
    cart_accel_noise: float = 0.06  # m/s^2 per sample
    link_accel_noise: float = 1.4  # rad/s^2 per sample (friction, model error, taps)
    command_delta_tol: float | None = None  # m/s^2 per tick; penalise u[k] - u[k-1]
    joint_torque_noise: float = 0.0  # N m, independent joint torque kicks per sample
    disturbance_time_constant: float | None = None  # s, leaky joint torque estimates
    disturbance_torque_noise: float = 0.0  # N m per sample, torque-state random walk
    design_model_lag: float | None = None  # s; controller model only, never the plant
    model_servo_dynamics: bool = False  # design with the velocity PI loop and cart reaction
    angle_integral_time_constant: float | None = 0.125  # s; leaky absolute-angle integrals
    angle_integral_tol: float = 0.001  # rad s, low-frequency attitude cost
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive = (
            "cart_position_tol",
            "cart_velocity_tol",
            "angle_tol",
            "rate_tol",
            "accel_tol",
            "joint_angle_tol",
            "joint_rate_tol",
            "command_delta_tol",
            "disturbance_time_constant",
            "design_model_lag",
            "angle_integral_time_constant",
            "angle_integral_tol",
        )
        for name in positive:
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive, or None if optional")
        for name in (
            "cart_accel_noise",
            "link_accel_noise",
            "joint_torque_noise",
            "disturbance_torque_noise",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.joint_angle_weights is not None:
            weights = np.asarray(self.joint_angle_weights, dtype=float)
            if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("joint_angle_weights must be finite and nonnegative")
            object.__setattr__(self, "joint_angle_weights", tuple(weights))


def default_config(configuration: str | None = None) -> ControllerConfig:
    """Default tuning for an equilibrium (None or all "U" = upright).

    Configurations with hanging links have lightly damped swinging modes coupled
    to the inverted ones. A stiffer joint-bend penalty, a faster angle estimate
    and a weaker low-frequency attitude cost lift the hardest ones (DDUU, DUUU)
    from about 40-80% to about 85-95% survival in check_robustness.py; with the
    upright tuning DDUU falls even without model mismatch.
    """
    if configuration is None or "D" not in configuration.upper():
        return ControllerConfig()
    return ControllerConfig(joint_angle_tol=0.005, link_accel_noise=2.5, angle_integral_tol=0.003)


def _c2d(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    nx = a.shape[0]
    m = np.zeros((nx + b.shape[1], nx + b.shape[1]))
    m[:nx, :nx] = a
    m[:nx, nx:] = b
    e = expm(m * dt)
    return e[:nx, :nx], e[:nx, nx:]


class LQRController:
    def __init__(
        self,
        rig: RigParams,
        config: ControllerConfig | None = None,
        configuration: str | None = None,
    ) -> None:
        self.rig = rig
        self.n = rig.n_links
        self.configuration = parse_configuration(configuration, rig.n_links)
        self.config = config or default_config(self.configuration)
        self.theta_eq = equilibrium_angles(self.configuration, rig.n_links)
        self.delay = rig.timing.compute_delay_steps
        self.dt = rig.control_dt
        self._design()
        self.z = np.zeros(self.nz)

    # ------------------------------------------------------------------ design
    def _design(self) -> None:
        n, d, cfg = self.n, self.delay, self.config
        design_rig = self.rig
        if cfg.design_model_lag is not None:
            design_rig = replace(
                self.rig, drive=replace(self.rig.drive, model_lag=cfg.design_model_lag)
            )
        a_c, b_c = state_space(design_rig, self.configuration)
        mass = linearized_mass(design_rig, self.configuration)
        moment = horizontal_moments(design_rig, self.configuration)
        if cfg.model_servo_dynamics:
            # Linearise the same *identified* velocity servo, including the
            # pendulum reaction on the cart. Extra state e_v = v_setpoint - v.
            # Differentiating M_eff*a = kp*e_v + ki*integral(e_v)
            #                         - cart_viscous*v - b.T*(K*theta + D*omega)
            # avoids differentiating measurements or adding runtime logic.
            old_nx = a_c.shape[0]
            servo_a = np.zeros((old_nx + 1, old_nx + 1))
            servo_a[:old_nx, :old_nx] = a_c
            stiffness = a_c[3 + n :, 3 : 3 + n]
            damping = a_c[3 + n :, 3 + n :]
            accel_coupling = a_c[3 + n :, 2]
            effective_mass = (
                design_rig.cart.mass
                + sum(link.mass for link in design_rig.links)
                + moment @ accel_coupling
            )
            bandwidth = 1 / design_rig.drive.model_lag
            kp = (design_rig.cart.mass + sum(link.mass for link in design_rig.links)) * bandwidth
            ki = kp * bandwidth / 4
            servo_a[2] = 0.0
            servo_a[2, 2] = (
                -(kp + design_rig.cart.viscous + moment @ damping @ accel_coupling) / effective_mass
            )
            servo_a[2, 3 : 3 + n] = -moment @ damping @ stiffness / effective_mass
            servo_a[2, 3 + n : old_nx] = -moment @ (stiffness + damping @ damping) / effective_mass
            servo_a[2, -1] = ki / effective_mass
            servo_a[-1, 2] = -1.0
            a_c = servo_a
            b_c = np.zeros((old_nx + 1, 1))
            b_c[2, 0] = kp / effective_mass
            b_c[-1, 0] = 1.0
        physical_nx = a_c.shape[0]
        integral_start = a_c.shape[0]
        if cfg.angle_integral_time_constant is not None:
            augmented = np.zeros((integral_start + n, integral_start + n))
            augmented[:integral_start, :integral_start] = a_c
            augmented[integral_start:, 3 : 3 + n] = np.eye(n)
            augmented[integral_start:, integral_start:] = (
                -np.eye(n) / cfg.angle_integral_time_constant
            )
            a_c = augmented
            b_c = np.vstack((b_c, np.zeros((n, 1))))
        torque_start = a_c.shape[0]
        bend = np.eye(n) - np.eye(n, k=-1)
        # Hinge disturbance model in angular coordinates. Unmodelled reaction
        # on the cart/drive is covered separately by cart acceleration noise.
        torque_input = np.zeros((torque_start, n))
        torque_input[3 + n : 3 + 2 * n] = np.linalg.solve(mass, bend.T)
        if cfg.disturbance_time_constant is not None:
            augmented = np.zeros((torque_start + n, torque_start + n))
            augmented[:torque_start, :torque_start] = a_c
            augmented[:torque_start, torque_start:] = torque_input
            augmented[torque_start:, torque_start:] = -np.eye(n) / cfg.disturbance_time_constant
            a_c = augmented
            b_c = np.vstack((b_c, np.zeros((n, 1))))
        ad, bd = _c2d(a_c, b_c, self.dt)
        nx = ad.shape[0]
        # A rate cost uses the existing newest queue entry when there is delay.
        # With zero delay, retain one extra deterministic previous-command state.
        remember_command = int(d == 0 and cfg.command_delta_tol is not None)
        nz = nx + d + remember_command
        az = np.zeros((nz, nz))
        bz = np.zeros((nz, 1))
        az[:nx, :nx] = ad
        if d == 0:
            bz[:nx] = bd
            if remember_command:
                bz[-1, 0] = 1.0
        else:
            az[:nx, nx : nx + 1] = bd  # oldest in-flight command acts now
            for k in range(d - 1):
                az[nx + k, nx + k + 1] = 1.0
            bz[nz - 1, 0] = 1.0  # newest command enters the queue
        self.nx, self.nz, self.az, self.bz = nx, nz, az, bz
        self.physical_nx = physical_nx
        layout = ["x - x_ref", "v", "a", "theta_1..n - theta_eq", "omega_1..n"]
        if cfg.model_servo_dynamics:
            layout.append("servo_velocity_error")
        if cfg.angle_integral_time_constant is not None:
            layout.append("leaky_angle_integrals_1..n")
        if cfg.disturbance_time_constant is not None:
            layout.append("joint_torques_1..n")
        if d:
            layout.append("commands_oldest..newest")
        elif remember_command:
            layout.append("previous_command")
        self.state_layout = ", ".join(layout)

        q = np.zeros(nz)
        q[0] = 1 / cfg.cart_position_tol**2
        q[1] = 1 / cfg.cart_velocity_tol**2
        q[3 : 3 + n] = 1 / cfg.angle_tol**2
        q[3 + n : 3 + 2 * n] = 1 / cfg.rate_tol**2
        if cfg.angle_integral_time_constant is not None:
            q[integral_start : integral_start + n] = 1 / cfg.angle_integral_tol**2
        q_mat = np.diag(q)
        if cfg.joint_angle_tol:
            weights = (
                np.ones(n)
                if cfg.joint_angle_weights is None
                else np.asarray(cfg.joint_angle_weights)
            )
            if weights.shape != (n,):
                raise ValueError("joint_angle_weights must contain one weight per joint")
            q_mat[3 : 3 + n, 3 : 3 + n] += bend.T @ np.diag(weights) @ bend / cfg.joint_angle_tol**2
        if cfg.joint_rate_tol:
            q_mat[3 + n : 3 + 2 * n, 3 + n : 3 + 2 * n] += bend.T @ bend / cfg.joint_rate_tol**2
        r = np.array([[1 / cfg.accel_tol**2]])
        cross = np.zeros((nz, 1))
        if cfg.command_delta_tol is not None:
            delta_weight = 1 / cfg.command_delta_tol**2
            q_mat[-1, -1] += delta_weight
            cross[-1, 0] = -delta_weight
            r[0, 0] += delta_weight
        p = solve_discrete_are(az, bz, q_mat, r, s=cross)
        self.k = np.linalg.solve(r + bz.T @ p @ bz, bz.T @ p @ az + cross.T)

        c = np.zeros((1 + n, nz))
        c[0, 0] = 1.0
        c[1:, 3 : 3 + n] = np.eye(n)
        s = self.rig.sensors
        quant = 2 * np.pi / 2**s.encoder_bits
        enc_var = quant**2 / 12 + s.encoder_noise**2
        # Absolute angle i sums i encoder readings, so its noise grows with i.
        meas_cov = np.zeros((1 + n, 1 + n))
        meas_cov[0, 0] = s.cart_resolution**2 / 12 + s.cart_noise**2
        meas_cov[1:, 1:] = enc_var * np.minimum.outer(np.arange(1, n + 1), np.arange(1, n + 1))
        w = np.zeros((nz, nz))
        w[:physical_nx, :physical_nx] = np.eye(physical_nx) * 1e-12
        w[2, 2] += cfg.cart_accel_noise**2
        w[3 + n : 3 + 2 * n, 3 + n : 3 + 2 * n] += np.eye(n) * (cfg.link_accel_noise * self.dt) ** 2
        _, torque_discrete = _c2d(
            a_c, np.vstack((torque_input, np.zeros((nx - torque_start, n)))), self.dt
        )
        w[:nx, :nx] += torque_discrete @ torque_discrete.T * cfg.joint_torque_noise**2
        if cfg.disturbance_time_constant is not None:
            w[torque_start:nx, torque_start:nx] += np.eye(n) * cfg.disturbance_torque_noise**2
        sp = solve_discrete_are(az.T, c.T, w, meas_cov)
        self.l = sp @ c.T @ np.linalg.inv(c @ sp @ c.T + meas_cov)  # (nz, 1+n)
        self.c = c
        self.closed_loop_radius = float(np.abs(np.linalg.eigvals(az - bz @ self.k)).max())
        self.observer_radius = float(np.abs(np.linalg.eigvals(az - self.l @ c @ az)).max())

    # ----------------------------------------------------------------- runtime
    def _y(self, meas: Measurement, x_ref: float) -> np.ndarray:
        theta = relative_to_absolute(np.asarray(meas.joint_angles, dtype=float))
        return np.concatenate([[meas.cart_position - x_ref], wrap_angle(theta - self.theta_eq)])

    def reset(self, meas: Measurement, x_ref: float = 0.0) -> None:
        """Initialise the estimate from one reading, at rest, with nothing in flight."""
        self.z = np.zeros(self.nz)
        y = self._y(meas, x_ref)
        self.z[0] = y[0]
        self.z[3 : 3 + self.n] = y[1:]
        self._x_ref = x_ref

    def step(self, meas: Measurement, x_ref: float = 0.0) -> float:
        """One control period: correct with the new reading, compute, predict."""
        # A reference change shifts the position coordinate of the estimate.
        self.z[0] -= x_ref - getattr(self, "_x_ref", 0.0)
        self._x_ref = x_ref
        y = self._y(meas, x_ref)
        self.z = self.z + self.l @ (y - self.c @ self.z)
        u = float(-(self.k @ self.z)[0])
        u = float(np.clip(u, -self.rig.drive.accel_limit, self.rig.drive.accel_limit))
        self.z = self.az @ self.z + self.bz[:, 0] * u
        return u

    @property
    def theta_estimate(self) -> np.ndarray:
        """Estimated absolute link angles."""
        return self.z[3 : 3 + self.n] + self.theta_eq

    # ----------------------------------------------------------------- export
    def to_c_header(self) -> str:
        """Gains and matrices for the firmware (row-major float arrays)."""

        def arr(name: str, m: np.ndarray) -> str:
            flat = ", ".join(f"{v:.9e}f" for v in np.asarray(m).ravel())
            return f"static const float {name}[{m.size}] = {{{flat}}};"

        lines = [
            "// Generated by quad_pendulum.controller.LQRController.to_c_header()",
            f"// configuration (bottom link first, U = up, D = down): {self.configuration}",
            f"// z = [{self.state_layout}]",
            "// y = [x - x_ref, wrap(theta_i - QP_THETA_EQ[i])], theta_i = sum of encoder angles",
            "// reset: zero all states; initialise z[0] and z[3..3+n-1] from y",
            "// reference change: z[0] -= x_ref - previous_x_ref before correction",
            "// each period: z += L (y - C z); u = clamp(-K z); z = A z + B u",
            "#pragma once",
            f"#define QP_NZ {self.nz}",
            f"#define QP_NY {1 + self.n}",
            f"#define QP_DT {self.dt:.9e}f",
            f"#define QP_ACCEL_LIMIT {self.rig.drive.accel_limit:.9e}f",
            f'#define QP_CONFIGURATION "{self.configuration}"',
            arr("QP_THETA_EQ", self.theta_eq),
            arr("QP_K", self.k),
            arr("QP_L", self.l),
            arr("QP_A", self.az),
            arr("QP_B", self.bz),
            arr("QP_C", self.c),
        ]
        return "\n".join(lines) + "\n"
