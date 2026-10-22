import re
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from quad_pendulum.controller import ControllerConfig, LQRController, default_config
from quad_pendulum.linear import unstable_poles
from quad_pendulum.params import (
    Measurement,
    RigParams,
    absolute_to_relative,
    all_configurations,
    equilibrium_angles,
)


def test_default_rig_is_open_loop_unstable_with_four_modes():
    poles = unstable_poles(RigParams())
    assert len(poles) == 4
    assert np.all(poles > 0)


def test_lqr_and_observer_are_stable():
    ctl = LQRController(RigParams())
    assert ctl.closed_loop_radius < 1.0
    assert ctl.observer_radius < 1.0


def test_single_link_controller_drives_cart_under_the_lean():
    # With one link the sign is unambiguous: leaning toward +x needs +x acceleration.
    # (With four links the optimal first move can go the other way to bend the chain.)
    rig = RigParams().with_links([0.3], [0.27])
    ctl = LQRController(rig)
    ctl.reset(Measurement(0.0, 0.0, np.zeros(1)))
    assert ctl.step(Measurement(0.001, 0.0, np.array([0.01]))) > 0


def test_command_respects_accel_limit():
    rig = RigParams()
    ctl = LQRController(rig)
    ctl.reset(Measurement(0.0, 0.0, np.zeros(rig.n_links)))
    u = ctl.step(Measurement(0.001, 0.0, np.array([0.3, 0.0, 0.0, 0.0])))
    assert abs(u) <= rig.drive.accel_limit


def test_c_header_export_has_all_matrices():
    header = LQRController(RigParams()).to_c_header()
    for name in ("QP_K", "QP_L", "QP_A", "QP_B", "QP_C"):
        assert f"static const float {name}" in header


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_rate_cost_keeps_the_saturated_previous_command_exact(delay):
    rig = RigParams()
    rig = replace(rig, timing=replace(rig.timing, compute_delay_steps=delay))
    ctl = LQRController(rig, ControllerConfig(command_delta_tol=0.3))
    ctl.reset(Measurement(0.0, 0.0, np.zeros(rig.n_links)))
    assert ctl.nz == ctl.nx + max(1, delay)
    np.testing.assert_allclose(ctl.l[ctl.nx :], 0.0, atol=1e-12)
    saturated = False
    for i in range(20):
        command = ctl.step(Measurement(i * rig.control_dt, 0.0, np.full(4, 0.4)))
        saturated |= abs(command) == rig.drive.accel_limit
        assert ctl.z[-1] == command
    assert saturated
    assert ctl.closed_loop_radius < 1.0


def test_rate_gain_minimises_the_cost_of_command_changes():
    # Independently form the cross term for (u - u_previous)^2. This catches
    # an ordinary effort penalty or a post-hoc smoother masquerading as rate LQR.
    cfg = ControllerConfig(command_delta_tol=0.2)
    ctl = LQRController(RigParams(), cfg)
    n = ctl.n
    q = np.zeros((ctl.nz, ctl.nz))
    q[0, 0] = cfg.cart_position_tol**-2
    q[1, 1] = cfg.cart_velocity_tol**-2
    q[3 : 3 + n, 3 : 3 + n] = np.eye(n) / cfg.angle_tol**2
    q[3 + n : 3 + 2 * n, 3 + n : 3 + 2 * n] = np.eye(n) / cfg.rate_tol**2
    bend = np.eye(n) - np.eye(n, k=-1)
    if cfg.joint_angle_tol is not None:
        weights = np.ones(n) if cfg.joint_angle_weights is None else cfg.joint_angle_weights
        q[3 : 3 + n, 3 : 3 + n] += bend.T @ np.diag(weights) @ bend / cfg.joint_angle_tol**2
    if cfg.angle_integral_time_constant is not None:
        start = ctl.physical_nx
        q[start : start + n, start : start + n] = np.eye(n) / cfg.angle_integral_tol**2
    rho = cfg.command_delta_tol**-2
    q[-1, -1] = rho
    r = np.array([[cfg.accel_tol**-2 + rho]])
    cross = np.zeros((ctl.nz, 1))
    cross[-1] = -rho
    p = solve_discrete_are(ctl.az, ctl.bz, q, r, s=cross)
    z = np.random.default_rng(4).normal(0.0, 0.01, ctl.nz)
    u = float(-(ctl.k @ z)[0])
    # First derivative of one-stage cost plus the Riccati value-to-go is zero.
    gradient = r[0, 0] * u + float((cross.T @ z)[0])
    gradient += float((ctl.bz.T @ p @ (ctl.az @ z + ctl.bz[:, 0] * u))[0])
    assert abs(gradient) < 1e-5


def test_augmented_observer_and_design_lag_preserve_the_hardware():
    rig = RigParams()
    cfg = ControllerConfig(
        command_delta_tol=0.3,
        joint_torque_noise=0.001,
        disturbance_time_constant=0.1,
        disturbance_torque_noise=1e-5,
        angle_integral_time_constant=0.2,
        design_model_lag=0.003,
    )
    ctl = LQRController(rig, cfg)
    assert ctl.rig is rig
    assert rig.drive.model_lag == 0.005
    assert ctl.nz == 3 + 4 * rig.n_links + rig.timing.compute_delay_steps
    assert ctl.closed_loop_radius < 1.0
    assert ctl.observer_radius < 1.0
    assert np.all(np.isfinite(ctl.k))
    assert np.all(np.isfinite(ctl.l))
    ctl.reset(Measurement(0.0, 0.0, np.zeros(4)))
    ctl.step(Measurement(0.001, 0.0, np.array([0.001, -0.001, 0.001, 0.0])))
    assert np.any(ctl.z[ctl.physical_nx : ctl.nx] != 0)
    ctl.reset(Measurement(0.0, 0.0, np.zeros(4)))
    np.testing.assert_array_equal(ctl.z, 0.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"command_delta_tol": 0},
        {"design_model_lag": -1},
        {"link_accel_noise": -1},
        {"disturbance_time_constant": float("nan")},
        {"joint_torque_noise": float("inf")},
        {"joint_angle_weights": [1, -1, 1, 1]},
    ],
)
def test_invalid_tuning_is_rejected(overrides):
    with pytest.raises(ValueError):
        ControllerConfig(**overrides)


def test_joint_weight_count_matches_the_chain():
    with pytest.raises(ValueError, match="one weight per joint"):
        LQRController(RigParams(), ControllerConfig(joint_angle_weights=(1.0, 2.0)))


@pytest.mark.parametrize("delay", [0, 1, 2])
@pytest.mark.parametrize("servo", [False, True])
def test_exported_float_c_executes_the_same_controller(tmp_path, delay, servo):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("a C compiler is needed to execute exported firmware matrices")
    rig = RigParams()
    rig = replace(rig, timing=replace(rig.timing, compute_delay_steps=delay))
    cfg = ControllerConfig(
        command_delta_tol=0.3,
        joint_torque_noise=0.001,
        disturbance_time_constant=0.1,
        disturbance_torque_noise=1e-5,
        angle_integral_time_constant=0.2,
        design_model_lag=0.003,
        model_servo_dynamics=servo,
    )
    ctl = LQRController(rig, cfg)
    header = ctl.to_c_header()
    floats = re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]\d+)?f\b", header)
    assert floats and all("e" in value for value in floats)
    (tmp_path / "gains.h").write_text(header)
    source = tmp_path / "runner.c"
    source.write_text(r"""
#include <stdio.h>
#include <math.h>
#include "gains.h"
int main(void) {
    float z[QP_NZ] = {0}, previous_reference = 0, reference, y[QP_NY];
    while (scanf("%f %f %f %f %f %f", &reference, y, y+1, y+2, y+3, y+4) == 6) {
        z[0] -= reference - previous_reference;
        previous_reference = reference;
        y[0] -= reference;
        float innovation[QP_NY], next[QP_NZ];
        for (int i = 0; i < QP_NY; ++i) {
            innovation[i] = y[i];
            for (int j = 0; j < QP_NZ; ++j) innovation[i] -= QP_C[i*QP_NZ+j]*z[j];
        }
        for (int i = 0; i < QP_NZ; ++i)
            for (int j = 0; j < QP_NY; ++j) z[i] += QP_L[i*QP_NY+j]*innovation[j];
        float u = 0;
        for (int i = 0; i < QP_NZ; ++i) u -= QP_K[i]*z[i];
        u = fminf(QP_ACCEL_LIMIT, fmaxf(-QP_ACCEL_LIMIT, u));
        for (int i = 0; i < QP_NZ; ++i) {
            next[i] = QP_B[i]*u;
            for (int j = 0; j < QP_NZ; ++j) next[i] += QP_A[i*QP_NZ+j]*z[j];
        }
        for (int i = 0; i < QP_NZ; ++i) z[i] = next[i];
        printf("%.9g\n", u);
    }
    return 0;
}
""")
    binary = tmp_path / "runner"
    subprocess.run(
        [cc, "-std=c11", "-Wall", "-Werror", str(source), "-lm", "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    samples = np.random.default_rng(32).normal(0, 2e-4, (300, 5))
    commands, lines = [], []
    for i, sample in enumerate(samples):
        reference = 0.2 if 100 <= i < 200 else 0.0
        commands.append(ctl.step(Measurement(i * rig.control_dt, sample[0], sample[1:]), reference))
        y = np.r_[reference, sample[0], np.cumsum(sample[1:])]
        lines.append(" ".join(f"{value:.17g}" for value in y))
    result = subprocess.run(
        [str(binary)], input="\n".join(lines) + "\n", text=True, capture_output=True, check=True
    )
    np.testing.assert_allclose(
        np.fromstring(result.stdout, sep="\n"), commands, atol=0.01, rtol=1e-3
    )


@pytest.mark.parametrize("cfg", all_configurations(4))
def test_every_configuration_has_stable_lqr_and_observer(cfg):
    ctl = LQRController(RigParams(), configuration=cfg)
    assert ctl.configuration == cfg
    np.testing.assert_allclose(ctl.theta_eq, equilibrium_angles(cfg, 4))
    assert ctl.closed_loop_radius < 1.0
    assert ctl.observer_radius < 1.0


def test_default_config_depends_on_configuration():
    assert default_config(None) == ControllerConfig()
    assert default_config("UUUU") == ControllerConfig()
    assert default_config("UUUD").joint_angle_tol == 0.005
    assert default_config("uudd") == default_config("UUUD")


@pytest.mark.parametrize("cfg", ["UUUU", "UUUD", "DDUU", "DUDU", "DDDD"])
def test_measurement_at_equilibrium_gives_zero_command(cfg):
    rig = RigParams()
    ctl = LQRController(rig, configuration=cfg)
    theta = ctl.theta_eq
    meas = Measurement(0.0, 0.0, absolute_to_relative(theta))
    ctl.reset(meas)
    assert abs(ctl.step(meas)) < 1e-9
    np.testing.assert_allclose(ctl.theta_estimate, theta, atol=1e-9)


def test_encoder_wrap_by_two_pi_does_not_change_command():
    rig = RigParams()
    cfg = "UUUD"
    plain = absolute_to_relative(equilibrium_angles(cfg, 4) + np.array([0.01, -0.02, 0.015, 0.005]))
    wrapped = plain.copy()
    wrapped[2] += 2 * np.pi
    commands, estimates = [], []
    for q in (plain, wrapped):
        ctl = LQRController(rig, configuration=cfg)
        ctl.reset(Measurement(0.0, 0.01, q))
        commands.append(ctl.step(Measurement(0.001, 0.01, q)))
        estimates.append(ctl.theta_estimate)
    assert abs(commands[0]) > 1e-3
    assert commands[1] == pytest.approx(commands[0], abs=1e-9)
    np.testing.assert_allclose(estimates[1], estimates[0], atol=1e-9)


def test_c_header_names_configuration_and_equilibrium():
    header = LQRController(RigParams(), configuration="UUUD").to_c_header()
    assert 'QP_CONFIGURATION "UUUD"' in header
    assert "QP_THETA_EQ" in header
