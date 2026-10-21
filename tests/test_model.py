from dataclasses import replace

import mujoco
import numpy as np
import pytest

from quad_pendulum.linear import angular_acceleration
from quad_pendulum.model import make_rig_xml
from quad_pendulum.params import (
    JointParams,
    LinkParams,
    RigParams,
    absolute_to_relative,
    relative_to_absolute,
)


@pytest.mark.parametrize("n", [1, 4, 6])
def test_xml_encodes_measured_masses_inertias_and_com(n):
    rig = replace(
        RigParams(),
        links=tuple(LinkParams(0.2 + 0.03 * i, 0.15 + 0.04 * i, 0.01 * i) for i in range(n)),
    )
    model = mujoco.MjModel.from_xml_string(make_rig_xml(rig))
    assert model.nq == model.nv == n + 1
    assert model.nu == 1
    assert model.body("cart").mass[0] == pytest.approx(rig.cart.mass)
    assert sum(model.body(f"link{i + 1}").mass[0] for i in range(n)) == pytest.approx(
        sum(link.mass for link in rig.links)
    )
    for i, link in enumerate(rig.links):
        body = model.body(f"link{i + 1}")
        np.testing.assert_allclose(body.ipos, [0, 0, link.com])
        np.testing.assert_allclose(
            body.inertia, [link.inertia_com, link.inertia_com, link.inertia_com * 0.001]
        )
        joint = model.joint(f"hinge{i + 1}")
        np.testing.assert_array_equal(joint.axis, [0, 1, 0])
        assert model.dof_frictionloss[i + 1] == rig.joints.coulomb
        assert model.dof_damping[i + 1] == rig.joints.viscous
    np.testing.assert_array_equal(model.geom_contype, 0)
    np.testing.assert_array_equal(model.geom_conaffinity, 0)
    np.testing.assert_array_equal(model.dof_armature, 0)
    assert model.dof_frictionloss[0] == rig.cart.friction
    assert model.dof_damping[0] == rig.cart.viscous
    np.testing.assert_allclose(
        model.jnt_range[0], [-rig.cart.rail_half_length - 0.05, rig.cart.rail_half_length + 0.05]
    )
    assert model.jnt_limited[0]
    np.testing.assert_allclose(
        model.actuator_ctrlrange[0], [-rig.drive.force_limit, rig.drive.force_limit]
    )
    assert model.actuator_gear[0, 0] == 1
    assert model.opt.timestep == rig.timing.physics_dt
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -rig.gravity])
    assert model.camera("side").id >= 0


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("cart_accel", [0.0, 2.3, -1.7])
@pytest.mark.parametrize("moving", [False, True])
def test_angular_acceleration_matches_independent_analytic_model(n, cart_accel, moving):
    base = RigParams()
    rig = replace(
        base,
        links=tuple(LinkParams(0.21 + 0.03 * i, 0.18 + 0.02 * i, 0.04) for i in range(n)),
        joints=JointParams(coulomb=0.0, viscous=0.0),
        cart=replace(base.cart, friction=0.0, viscous=0.0),
    )
    model = mujoco.MjModel.from_xml_string(make_rig_xml(rig))
    data = mujoco.MjData(model)
    theta = np.linspace(0.012, -0.009, n)
    omega = np.linspace(-0.4, 0.7, n) if moving else np.zeros(n)
    if moving:
        theta *= 12  # Also exercise nonlinear Coriolis/centrifugal terms.
    data.qpos[1:] = absolute_to_relative(theta)
    data.qvel[1:] = absolute_to_relative(omega)
    mujoco.mj_forward(model, data)

    # Find the slider force that imposes a prescribed acceleration, using only
    # MuJoCo's mass matrix and bias. No angular forcing or analytic terms enter.
    mass = np.empty((model.nv, model.nv))
    for i, basis in enumerate(np.eye(model.nv)):
        mujoco.mj_mulM(model, data, mass[i], basis)
    target = np.empty(model.nv)
    target[0] = cart_accel
    target[1:] = np.linalg.solve(mass[1:, 1:], -data.qfrc_bias[1:] - mass[1:, 0] * cart_accel)
    data.qfrc_applied[0] = mass[0] @ target + data.qfrc_bias[0]
    mujoco.mj_forward(model, data)

    assert data.qacc[0] == pytest.approx(cart_accel, abs=1e-10)
    np.testing.assert_allclose(
        relative_to_absolute(data.qacc[1:]),
        angular_acceleration(rig, theta, omega, cart_accel),
        rtol=0.01,
        atol=1e-9,
    )


def test_empty_chain_rejected():
    with pytest.raises(ValueError, match="at least one link"):
        make_rig_xml(replace(RigParams(), links=()))
