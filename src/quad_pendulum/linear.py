"""Analytic dynamics of the n-link chain on an acceleration-driven cart.

This is independent of MuJoCo on purpose: the controller is designed from this
model (as it would be on the real rig), and the tests check it against the
MuJoCo plant.

With cart acceleration a as an input and absolute link angles theta, the
equations of motion are

    M(theta) theta'' + C(theta, theta') + D theta' = g b sin(theta) - b cos(theta) a

where the mass terms come from summing every rod and point mass.

Any equilibrium with each link straight up or straight down can be linearised:
with signs s_i = cos(theta*_i) = +1 (up) or -1 (down) and S = diag(s), the
deviation dynamics are

    (S Mp S + I) dtheta'' + D dtheta' = g diag(b s) dtheta - (S b) a
"""

from __future__ import annotations

import numpy as np

from quad_pendulum.params import RigParams, equilibrium_angles


def mass_terms(rig: RigParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Mp, I, b).

    Mp[i, j] = sum_k m_k c_ki c_kj  over every mass, where c_ki is the horizontal
    lever arm of mass k about hinge i when everything is upright.
    I[i] = rod inertia about its own centre of mass.
    b[i] = sum_k m_k c_ki (first moment of everything above hinge i).
    """
    n = rig.n_links
    lengths = np.array([link.length for link in rig.links])
    masses: list[tuple[float, np.ndarray]] = []
    inertia = np.zeros(n)
    for i, link in enumerate(rig.links):
        c = np.zeros(n)
        c[:i] = lengths[:i]
        c[i] = link.length / 2
        masses.append((link.rod_mass, c))
        inertia[i] = link.rod_mass * link.length**2 / 12
        if link.top_mass > 0:
            c = np.zeros(n)
            c[: i + 1] = lengths[: i + 1]
            masses.append((link.top_mass, c))
    mp = sum(m * np.outer(c, c) for m, c in masses)
    b = sum(m * c for m, c in masses)
    return mp, inertia, b


def joint_damping_matrix(rig: RigParams) -> np.ndarray:
    """Viscous hinge torques in absolute coordinates (joints act between links)."""
    n = rig.n_links
    # relative rates qdot = T theta_dot, torque on links = -T^T c T theta_dot
    t = np.eye(n) - np.eye(n, k=-1)
    return rig.joints.viscous * t.T @ t


def angular_acceleration(
    rig: RigParams, theta: np.ndarray, omega: np.ndarray, cart_accel: float
) -> np.ndarray:
    """Nonlinear theta'' for given state (no Coulomb friction)."""
    mp, inertia, b = mass_terms(rig)
    d = theta[:, None] - theta[None, :]
    m = mp * np.cos(d) + np.diag(inertia)
    rhs = (
        -(mp * np.sin(d)) @ (omega**2)
        + rig.gravity * b * np.sin(theta)
        - b * np.cos(theta) * cart_accel
        - joint_damping_matrix(rig) @ omega
    )
    return np.linalg.solve(m, rhs)


def equilibrium_signs(rig: RigParams, configuration: str | None = None) -> np.ndarray:
    """cos(theta*) per link: +1 for an upright link, -1 for a hanging one."""
    return np.cos(equilibrium_angles(configuration, rig.n_links)).round()


def linearized_mass(rig: RigParams, configuration: str | None = None) -> np.ndarray:
    """Mass matrix M(theta*) in absolute coordinates at the equilibrium."""
    mp, inertia, _ = mass_terms(rig)
    s = equilibrium_signs(rig, configuration)
    return mp * np.outer(s, s) + np.diag(inertia)


def horizontal_moments(rig: RigParams, configuration: str | None = None) -> np.ndarray:
    """d(horizontal mass moment)/d(theta_i) at the equilibrium: b_i s_i."""
    _, _, b = mass_terms(rig)
    return b * equilibrium_signs(rig, configuration)


def linearized(
    rig: RigParams, configuration: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearisation about an equilibrium: dtheta'' = K dtheta + D omega + Bv a."""
    _, _, b = mass_terms(rig)
    s = equilibrium_signs(rig, configuration)
    minv = np.linalg.inv(linearized_mass(rig, configuration))
    k = minv @ np.diag(rig.gravity * b * s)
    d = -minv @ joint_damping_matrix(rig)
    bv = -minv @ (b * s)
    return k, d, bv


def unstable_poles(rig: RigParams, configuration: str | None = None) -> np.ndarray:
    """Open-loop unstable eigenvalues (rad/s), fastest first."""
    k, _, _ = linearized(rig, configuration)
    ev = np.linalg.eigvals(k).real
    return np.sort(np.sqrt(ev[ev > 0]))[::-1]


def state_space(
    rig: RigParams, configuration: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous model used for control design.

    State: [x, v, a, theta_1..n - theta*, omega_1..n]  (a = actual cart acceleration)
    Input: commanded acceleration. The drive is modelled as a first-order lag.
    """
    n = rig.n_links
    tau = rig.drive.model_lag
    k, d, bv = linearized(rig, configuration)
    nx = 3 + 2 * n
    a_mat = np.zeros((nx, nx))
    b_mat = np.zeros((nx, 1))
    a_mat[0, 1] = 1.0
    a_mat[1, 2] = 1.0
    a_mat[2, 2] = -1.0 / tau
    b_mat[2, 0] = 1.0 / tau
    a_mat[3 : 3 + n, 3 + n :] = np.eye(n)
    a_mat[3 + n :, 3 : 3 + n] = k
    a_mat[3 + n :, 3 + n :] = d
    a_mat[3 + n :, 2] = bv
    return a_mat, b_mat
