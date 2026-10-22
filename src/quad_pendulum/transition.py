"""Real-time transitions between up/down equilibria (swing-up, swing-down, flips).

Offline, `plan_transition` finds a feasible motion from one configuration to
another by direct collocation (CasADi + IPOPT, optional dependency): cart
position, velocity and acceleration, link angles and rates, with the cart
acceleration as a state and its rate (jerk) as the input, so the commanded
acceleration u = a + model_lag * jerk is consistent with the drive lag. Rail,
speed, command and belt-force limits are constraints. Only the knots are stored
(`Trajectory`, a small JSON file), shared by Python and the browser.

At run time `TransitionController` samples the knots at the control rate with
cubic Hermite interpolation, linearises the chain along that nominal motion,
and designs a time-varying LQR (backward Riccati from the stationary solution
at the goal) and a time-varying Kalman filter (forward Riccati, run online).
The state is the deviation from the nominal, with the same delay augmentation
as `LQRController`:

    z = [x - x_nom, v - v_nom, a - a_nom, theta - theta_nom, omega - omega_nom,
         commands in flight - nominal commands]

and each period: z += L_k (y - C z); u = u_nom + clamp(-K_k z); z = A_k z + B_k (u - u_nom).
The nominal command issued at step k is the one that acts at k + delay.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from quad_pendulum.linear import joint_damping_matrix, mass_terms
from quad_pendulum.params import (
    Measurement,
    RigParams,
    equilibrium_angles,
    parse_configuration,
    relative_to_absolute,
    wrap_angle,
)

# The planned library: data/transitions/<preset id>/ (see scripts/plan_transitions.py).
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "transitions"


@dataclass(frozen=True)
class PlanLimits:
    """Margins below the rig limits that the planned motion must respect."""

    rail: float = 0.5  # m, |x| (the end stops are at rail_half_length)
    velocity: float = 2.2  # m/s
    accel_command: float = 25.0  # m/s^2
    force: float = 120.0  # N at the belt
    link_rate: float = 25.0  # rad/s
    link_accel: float = 300.0  # rad/s^2 (keeps the collocation from exploiting the grid)


@dataclass
class Trajectory:
    """Collocation knots of one transition, evenly spaced over `duration`."""

    start: str
    goal: str
    duration: float
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    jerk: np.ndarray
    theta: np.ndarray  # (n, knots)
    omega: np.ndarray
    alpha: np.ndarray
    cost: float = 0.0
    limits: PlanLimits = field(default_factory=PlanLimits)

    @property
    def n_links(self) -> int:
        return self.theta.shape[0]

    def to_json(self) -> dict:
        def r(v):
            return np.round(np.asarray(v, dtype=float), 6).tolist()

        return {
            "start": self.start,
            "goal": self.goal,
            "duration": self.duration,
            "knots": int(self.x.size),
            "x": r(self.x),
            "v": r(self.v),
            "a": r(self.a),
            "jerk": r(self.jerk),
            "theta": r(self.theta),
            "omega": r(self.omega),
            "alpha": r(self.alpha),
            "cost": float(self.cost),
            "limits": asdict(self.limits),
        }

    @classmethod
    def from_json(cls, d: dict) -> Trajectory:
        arr = {k: np.asarray(d[k], dtype=float) for k in ("x", "v", "a", "jerk")}
        mat = {k: np.atleast_2d(np.asarray(d[k], dtype=float)) for k in ("theta", "omega", "alpha")}
        return cls(
            start=d["start"],
            goal=d["goal"],
            duration=float(d["duration"]),
            cost=float(d.get("cost", 0.0)),
            limits=PlanLimits(**d.get("limits", {})),
            **arr,
            **mat,
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        return cls.from_json(json.loads(Path(path).read_text()))

    def sample(self, dt: float, model_lag: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nominal (t, state, command) at the control rate.

        State rows are [x, v, a, theta_1..n, omega_1..n]. Each quantity is a cubic
        Hermite spline using its derivative at the knots; jerk is linear.
        """
        knots = self.x.size
        steps = round(self.duration / dt)
        t = np.arange(steps + 1) * dt
        h = self.duration / (knots - 1)
        s = np.clip(t / h, 0.0, knots - 1 - 1e-12)
        i = np.floor(s).astype(int)
        f = s - i

        def herm(p, m):
            p0, p1, m0, m1 = p[i], p[i + 1], m[i] * h, m[i + 1] * h
            f2, f3 = f * f, f * f * f
            return (
                (2 * f3 - 3 * f2 + 1) * p0
                + (f3 - 2 * f2 + f) * m0
                + (-2 * f3 + 3 * f2) * p1
                + (f3 - f2) * m1
            )

        n = self.n_links
        rows = [herm(self.x, self.v), herm(self.v, self.a), herm(self.a, self.jerk)]
        rows += [herm(self.theta[j], self.omega[j]) for j in range(n)]
        rows += [herm(self.omega[j], self.alpha[j]) for j in range(n)]
        state = np.vstack(rows)
        jerk = self.jerk[i] * (1 - f) + self.jerk[i + 1] * f
        return t, state, state[2] + model_lag * jerk


# ------------------------------------------------------------------ planning
def reversed_trajectory(traj: Trajectory) -> Trajectory:
    """The same motion played backwards (goal -> start).

    Apart from the small joint damping the chain's dynamics are time-reversible,
    so this is a near-feasible motion and an excellent initial guess for the
    opposite transition, which IPOPT often cannot find from a random guess."""
    theta = traj.theta[:, ::-1].copy()
    eq = equilibrium_angles(traj.goal, traj.n_links)[:, None]
    theta -= 2 * np.pi * np.round((theta[:, :1] - eq) / (2 * np.pi))  # start at 0 or pi exactly
    return Trajectory(
        start=traj.goal,
        goal=traj.start,
        duration=traj.duration,
        x=traj.x[::-1].copy(),
        v=-traj.v[::-1],
        a=traj.a[::-1].copy(),
        jerk=-traj.jerk[::-1],
        theta=theta,
        omega=-traj.omega[:, ::-1],
        alpha=traj.alpha[:, ::-1].copy(),
        cost=traj.cost,
        limits=traj.limits,
    )


def plan_transition(
    rig: RigParams,
    start: str,
    goal: str,
    duration: float,
    *,
    knots: int = 501,
    coarse_knots: int = 151,
    seed: int = 0,
    limits: PlanLimits | None = None,
    max_iter: int = 3000,
    initial: Trajectory | None = None,
) -> Trajectory | None:
    """Direct collocation (trapezoidal) from rest at `start` to rest at `goal`, cart
    centred at both ends. Solves on a coarse grid from a random guess, then refines;
    with `initial` (e.g. `reversed_trajectory` of the opposite transition) it
    solves the fine grid directly from that guess, stretched to `duration`.
    Returns None if IPOPT does not converge. Needs `casadi` (pip install casadi)."""
    import casadi as ca

    n = rig.n_links
    start = parse_configuration(start, n)
    goal = parse_configuration(goal, n)
    lim = limits or PlanLimits()
    mp, inertia, b = mass_terms(rig)
    damping = joint_damping_matrix(rig)
    total_mass = rig.cart.mass + sum(link.mass for link in rig.links)
    tau = rig.drive.model_lag
    th0 = equilibrium_angles(start, n)
    th1 = equilibrium_angles(goal, n)
    rng = np.random.default_rng(seed)

    def solve(count: int, guess) -> tuple[bool, tuple]:
        opti = ca.Opti()
        xs = opti.variable(3 + 2 * n, count)
        jerk = opti.variable(1, count)
        alpha = opti.variable(n, count)
        h = duration / (count - 1)
        cost = 0
        for k in range(count):
            th, om, a = xs[3 : 3 + n, k], xs[3 + n :, k], xs[2, k]
            for i in range(n):
                r = -rig.gravity * b[i] * ca.sin(th[i]) + b[i] * ca.cos(th[i]) * a
                for j in range(n):
                    d = th[i] - th[j]
                    r += (mp[i, j] * ca.cos(d) + (inertia[i] if i == j else 0)) * alpha[j, k]
                    r += mp[i, j] * ca.sin(d) * om[j] ** 2 + damping[i, j] * om[j]
                opti.subject_to(r == 0)
            u = a + tau * jerk[0, k]
            force = total_mass * a + rig.cart.viscous * xs[1, k]
            for i in range(n):
                force += b[i] * (ca.cos(th[i]) * alpha[i, k] - ca.sin(th[i]) * om[i] ** 2)
            opti.subject_to(opti.bounded(-lim.accel_command, u, lim.accel_command))
            opti.subject_to(opti.bounded(-lim.force, force, lim.force))
            weight = 0.5 if k in (0, count - 1) else 1.0
            cost += weight * h * (u**2 + 1e-4 * jerk[0, k] ** 2)
        for k in range(count - 1):

            def deriv(k):
                return ca.vertcat(xs[1, k], xs[2, k], jerk[0, k], xs[3 + n :, k], alpha[:, k])

            opti.subject_to(xs[:, k + 1] == xs[:, k] + h / 2 * (deriv(k) + deriv(k + 1)))
        opti.subject_to(opti.bounded(-lim.rail, xs[0, :], lim.rail))
        opti.subject_to(opti.bounded(-lim.velocity, xs[1, :], lim.velocity))
        opti.subject_to(opti.bounded(-lim.link_rate, xs[3 + n :, :], lim.link_rate))
        opti.subject_to(opti.bounded(-lim.link_accel, alpha, lim.link_accel))
        opti.subject_to(xs[:, 0] == np.r_[0.0, 0.0, 0.0, th0, np.zeros(n)])
        opti.subject_to(xs[:3, -1] == 0)
        opti.subject_to(xs[3 + n :, -1] == 0)
        opti.subject_to(jerk[0, 0] == 0)
        opti.subject_to(jerk[0, -1] == 0)
        for i in range(n):  # any number of whole turns is the same equilibrium
            opti.subject_to(ca.sin(xs[3 + i, -1]) == np.sin(th1[i]))
            opti.subject_to(ca.cos(xs[3 + i, -1]) == np.cos(th1[i]))
        opti.minimize(cost)
        gx, gj, ga = guess
        opti.set_initial(xs, gx)
        opti.set_initial(jerk, gj)
        opti.set_initial(alpha, ga)
        opti.solver(
            "ipopt",
            {"print_time": False},
            {"max_iter": max_iter, "print_level": 0, "tol": 1e-6, "sb": "yes"},
        )
        try:
            sol = opti.solve()
        except RuntimeError:
            return False, ()
        return True, (sol.value(xs), np.atleast_1d(sol.value(jerk)), np.atleast_2d(sol.value(alpha)), float(sol.value(cost)))

    if initial is not None:
        fine_t = np.linspace(0.0, 1.0, knots)
        init_t = np.linspace(0.0, 1.0, initial.x.size)
        scale = initial.duration / duration  # rates scale when the motion is stretched

        def stretch(m, power):
            return np.array([np.interp(fine_t, init_t, row) for row in np.atleast_2d(m)]) * scale**power

        gx = np.vstack([
            stretch(initial.x, 0), stretch(initial.v, 1), stretch(initial.a, 2),
            stretch(initial.theta, 0), stretch(initial.omega, 1),
        ])
        ok, fine = solve(knots, (gx, stretch(initial.jerk, 3), stretch(initial.alpha, 2)))
        if not ok:
            return None
        xs, jerk, alpha, cost = fine
        return Trajectory(
            start=start, goal=goal, duration=float(duration), x=xs[0], v=xs[1], a=xs[2],
            jerk=np.ravel(jerk), theta=xs[3 : 3 + n], omega=xs[3 + n :], alpha=alpha,
            cost=cost, limits=lim,
        )

    # Random guess: each flipping link turns half a revolution either way while
    # the cart sways; IPOPT finds the real motion from there.
    t = np.linspace(0.0, 1.0, coarse_knots)
    target = th0 + np.where(th1 != th0, rng.choice([-1.0, 1.0], n) * np.pi, 0.0)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)
    gx = np.zeros((3 + 2 * n, coarse_knots))
    gx[3 : 3 + n] = th0[:, None] + (target - th0)[:, None] * ease
    gx[3 + n :] = np.gradient(gx[3 : 3 + n], duration / (coarse_knots - 1), axis=1)
    gx[0] = 0.25 * rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * rng.integers(1, 4) * t)
    ok, coarse = solve(coarse_knots, (gx, np.zeros(coarse_knots), np.zeros((n, coarse_knots))))
    if not ok:
        return None
    fine_t = np.linspace(0.0, 1.0, knots)

    def resample(m):
        return np.array([np.interp(fine_t, t, row) for row in np.atleast_2d(m)])

    ok, fine = solve(knots, (resample(coarse[0]), resample(coarse[1]), resample(coarse[2])))
    if not ok:
        return None
    xs, jerk, alpha, cost = fine
    return Trajectory(
        start=start,
        goal=goal,
        duration=float(duration),
        x=xs[0],
        v=xs[1],
        a=xs[2],
        jerk=np.ravel(jerk),
        theta=xs[3 : 3 + n],
        omega=xs[3 + n :],
        alpha=alpha,
        cost=cost,
        limits=lim,
    )


# ----------------------------------------------------------- run-time design
@dataclass(frozen=True)
class TransitionConfig:
    """Time-varying LQR weights (Bryson-style tolerances) and filter noise."""

    cart_position_tol: float = 0.08  # m
    cart_velocity_tol: float = 1.0  # m/s
    angle_tol: float = 0.05  # rad, absolute angle deviation from the nominal
    rate_tol: float = 0.5  # rad/s
    joint_angle_tol: float | None = 0.01  # rad, bend at each joint
    accel_tol: float = 1.0  # m/s^2, feedback command effort
    cart_accel_noise: float = 0.06  # m/s^2 per sample
    link_accel_noise: float = 2.5  # rad/s^2 per sample
    fd_step: float = 1e-6  # finite-difference step for the Jacobians


def _accel_batch(rig: RigParams, theta, omega, cart_accel) -> np.ndarray:
    """`angular_acceleration` for many states at once (rows of theta/omega)."""
    mp, inertia, b = mass_terms(rig)
    d = theta[:, :, None] - theta[:, None, :]
    m = mp * np.cos(d) + np.diag(inertia)
    rhs = (
        -np.einsum("kij,kj->ki", mp * np.sin(d), omega**2)
        + rig.gravity * b * np.sin(theta)
        - b * np.cos(theta) * cart_accel[:, None]
        - omega @ joint_damping_matrix(rig).T
    )
    return np.linalg.solve(m, rhs[:, :, None])[:, :, 0]


def chain_jacobian(rig: RigParams, theta, omega, cart_accel, step: float = 1e-6) -> np.ndarray:
    """d(theta'')/d[a, theta, omega] by central differences, shape (..., n, 2n + 1).

    Accepts one state or a batch (theta/omega of shape (k, n), cart_accel (k,))."""
    theta = np.atleast_2d(theta)
    omega = np.atleast_2d(omega)
    cart_accel = np.atleast_1d(np.asarray(cart_accel, dtype=float))
    n = theta.shape[1]
    base = np.hstack([cart_accel[:, None], theta, omega])
    jac = np.zeros((base.shape[0], n, 2 * n + 1))
    for i in range(2 * n + 1):
        hi, lo = base.copy(), base.copy()
        hi[:, i] += step
        lo[:, i] -= step
        jac[:, :, i] = (
            _accel_batch(rig, hi[:, 1 : 1 + n], hi[:, 1 + n :], hi[:, 0])
            - _accel_batch(rig, lo[:, 1 : 1 + n], lo[:, 1 + n :], lo[:, 0])
        ) / (2 * step)
    return jac[0] if jac.shape[0] == 1 else jac


def dare_doubling(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Discrete Riccati solution by structure-preserving doubling (also in the web code)."""
    g = b @ np.linalg.solve(r, b.T)
    h = q.copy()
    eye = np.eye(a.shape[0])
    for _ in range(100):
        w = eye + g @ h
        a_w = np.linalg.solve(w.T, a.T).T  # a @ inv(w)
        g_next = g + a_w @ g @ a.T
        h_next = h + a.T @ h @ np.linalg.solve(w, a)
        a = a_w @ a
        done = np.abs(h_next - h).max() <= 1e-12 * max(1.0, np.abs(h_next).max())
        g, h = (g_next + g_next.T) / 2, (h_next + h_next.T) / 2
        if done:
            break
    return h


class TransitionController:
    """Time-varying LQR + Kalman filter tracking one `Trajectory` on the real-time loop."""

    def __init__(
        self, rig: RigParams, trajectory: Trajectory, config: TransitionConfig | None = None
    ) -> None:
        if trajectory.n_links != rig.n_links:
            raise ValueError("trajectory and rig have different numbers of links")
        self.rig = rig
        self.trajectory = trajectory
        self.config = config or TransitionConfig()
        self.n = rig.n_links
        self.delay = rig.timing.compute_delay_steps
        self.dt = rig.control_dt
        self.t, self.nominal, self.u_nominal = trajectory.sample(self.dt, rig.drive.model_lag)
        self.steps = self.t.size - 1
        d = self.delay
        # The command issued at step k acts at k + d; after the end it is zero.
        self.u_issue = np.r_[self.u_nominal[d:], np.zeros(d)][: self.steps]
        self._design()
        self.k = 0
        self.x_offset = 0.0
        self.z = np.zeros(self.nz)
        self.p = np.eye(self.nz) * 1e-6

    def _design(self) -> None:
        n, d, cfg, rig = self.n, self.delay, self.config, self.rig
        ns = 3 + 2 * n
        nz = ns + d
        tau = rig.drive.model_lag
        self.ns, self.nz = ns, nz
        self.az = np.zeros((self.steps, nz, nz))
        self.bz = np.zeros((self.steps, nz))
        nom = self.nominal[:, : self.steps]
        jacs = chain_jacobian(rig, nom[3 : 3 + n].T, nom[3 + n :].T, nom[2], cfg.fd_step)
        jacs = jacs.reshape(self.steps, n, 2 * n + 1)
        for k in range(self.steps):
            jac = jacs[k]
            m = np.zeros((ns + 1, ns + 1))
            m[0, 1] = m[1, 2] = 1.0
            m[2, 2] = -1.0 / tau
            m[2, ns] = 1.0 / tau
            m[3 : 3 + n, 3 + n : ns] = np.eye(n)
            m[3 + n : ns, 2] = jac[:, 0]
            m[3 + n : ns, 3:ns] = jac[:, 1:]
            e = expm(m * self.dt)
            self.az[k, :ns, :ns] = e[:ns, :ns]
            if d == 0:
                self.bz[k, :ns] = e[:ns, ns]
            else:
                self.az[k, :ns, ns] = e[:ns, ns]
                for i in range(d - 1):
                    self.az[k, ns + i, ns + i + 1] = 1.0
                self.bz[k, -1] = 1.0
        q = np.zeros(nz)
        q[0] = 1 / cfg.cart_position_tol**2
        q[1] = 1 / cfg.cart_velocity_tol**2
        q[3 : 3 + n] = 1 / cfg.angle_tol**2
        q[3 + n : ns] = 1 / cfg.rate_tol**2
        qm = np.diag(q)
        if cfg.joint_angle_tol:
            bend = np.eye(n) - np.eye(n, k=-1)
            qm[3 : 3 + n, 3 : 3 + n] += bend.T @ bend / cfg.joint_angle_tol**2
        r = np.array([[1 / cfg.accel_tol**2]])
        # Terminal cost: the stationary solution at the goal (the end of the nominal).
        p = dare_doubling(self.az[-1], self.bz[-1][:, None], qm, r)
        self.gains = np.zeros((self.steps, nz))
        for k in range(self.steps - 1, -1, -1):
            a, b = self.az[k], self.bz[k][:, None]
            gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
            self.gains[k] = gain[0]
            p = qm + a.T @ p @ (a - b @ gain)
            p = (p + p.T) / 2
        c = np.zeros((1 + n, nz))
        c[0, 0] = 1.0
        c[1:, 3 : 3 + n] = np.eye(n)
        s = rig.sensors
        enc = (2 * np.pi / 2**s.encoder_bits) ** 2 / 12 + s.encoder_noise**2
        meas_cov = np.zeros((1 + n, 1 + n))
        meas_cov[0, 0] = s.cart_resolution**2 / 12 + s.cart_noise**2
        meas_cov[1:, 1:] = enc * np.minimum.outer(np.arange(1, n + 1), np.arange(1, n + 1))
        w = np.eye(nz) * 1e-12
        w[2, 2] += cfg.cart_accel_noise**2
        w[3 + n : ns, 3 + n : ns] += np.eye(n) * (cfg.link_accel_noise * self.dt) ** 2
        self.c, self.meas_cov, self.process_cov = c, meas_cov, w

    # ----------------------------------------------------------------- runtime
    def begin(self, deviation: np.ndarray, in_flight: np.ndarray, x_offset: float = 0.0) -> None:
        """Start tracking. `deviation`: estimated physical state minus the start
        equilibrium (cart measured from the centre); `in_flight`: commands queued.
        The motion does not depend on where the cart is, so the nominal is shifted
        by `x_offset` (keep it within the planning margin of the rail)."""
        self.k = 0
        self.x_offset = float(x_offset)
        self.z = np.zeros(self.nz)
        self.z[: self.ns] = deviation[: self.ns]
        self.z[0] -= self.x_offset
        if self.delay:
            self.z[self.ns :] = np.asarray(in_flight, dtype=float) - self.u_nominal[: self.delay]
        self.p = np.eye(self.nz) * 1e-6

    @property
    def done(self) -> bool:
        return self.k >= self.steps

    @property
    def progress(self) -> float:
        return min(1.0, self.k / self.steps)

    def step(self, meas: Measurement) -> float:
        """One control period along the nominal; call until `done`."""
        k, n = min(self.k, self.steps - 1), self.n
        theta = relative_to_absolute(np.asarray(meas.joint_angles, dtype=float))
        s = self.nominal[:, k]
        y = np.r_[meas.cart_position - s[0] - self.x_offset, wrap_angle(theta - s[3 : 3 + n])]
        c, p = self.c, self.p
        gain = p @ c.T @ np.linalg.inv(c @ p @ c.T + self.meas_cov)
        self.z = self.z + gain @ (y - c @ self.z)
        p = (np.eye(self.nz) - gain @ c) @ p
        limit = self.rig.drive.accel_limit
        u = float(np.clip(self.u_issue[k] - self.gains[k] @ self.z, -limit, limit))
        a = self.az[k]
        self.z = a @ self.z + self.bz[k] * (u - self.u_issue[k])
        self.p = a @ p @ a.T + self.process_cov
        self.k += 1
        return u

    @property
    def theta_estimate(self) -> np.ndarray:
        k = min(self.k, self.steps)
        return self.nominal[3 : 3 + self.n, k] + self.z[3 : 3 + self.n]

    def handoff(self) -> np.ndarray:
        """Estimated physical state minus the goal equilibrium at the end, cart
        measured from the centre (the nominal ends at rest at the goal)."""
        dev = self.z[: self.ns].copy()
        dev[0] += self.x_offset
        return dev


class TransitionLibrary:
    """Planned transitions for one rig, stored as `<start>-<goal>.json` in a directory.

    `route` finds the quickest chain of stored transitions between two
    configurations (Dijkstra on duration), so a sparse library (swing-ups from
    all-down, swing-downs, single-link flips) still connects every pair.
    """

    def __init__(self, directory: Path | None = None, trajectories: list[Trajectory] = ()) -> None:
        self._paths: dict[tuple[str, str], Path] = {}
        self._loaded: dict[tuple[str, str], Trajectory] = {}
        self._durations: dict[tuple[str, str], float] = {}
        if directory is not None:
            index = Path(directory) / "index.json"
            entries = json.loads(index.read_text()) if index.exists() else []
            for e in entries:
                key = (e["start"], e["goal"])
                self._paths[key] = Path(directory) / e["file"]
                self._durations[key] = float(e["duration"])
        for traj in trajectories:
            self.add(traj)

    def add(self, trajectory: Trajectory) -> None:
        key = (trajectory.start, trajectory.goal)
        self._loaded[key] = trajectory
        self._durations[key] = trajectory.duration

    @property
    def edges(self) -> list[tuple[str, str]]:
        return sorted(self._durations)

    def get(self, start: str, goal: str) -> Trajectory:
        key = (start, goal)
        if key not in self._loaded:
            self._loaded[key] = Trajectory.load(self._paths[key])
        return self._loaded[key]

    def route(self, start: str, goal: str) -> list[tuple[str, str]] | None:
        """Quickest sequence of stored transitions from `start` to `goal`."""
        if start == goal:
            return []
        best = {start: 0.0}
        prev: dict[str, str] = {}
        frontier = [(0.0, start)]

        while frontier:
            cost, node = heapq.heappop(frontier)
            if node == goal:
                break
            if cost > best.get(node, np.inf):
                continue
            for (a, b), duration in self._durations.items():
                if a == node and cost + duration < best.get(b, np.inf):
                    best[b] = cost + duration
                    prev[b] = a
                    heapq.heappush(frontier, (best[b], b))
        if goal not in best:
            return None
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return list(pairwise(path))
