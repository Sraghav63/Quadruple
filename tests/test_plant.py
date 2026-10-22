from dataclasses import replace

import mujoco
import numpy as np
import pytest

from quad_pendulum.model import make_rig_xml
from quad_pendulum.params import RigParams, absolute_to_relative, equilibrium_angles
from quad_pendulum.plant import Plant


def held_upright_plant(rig):
    """A physical holding jig: equality constraints keep the hinges near zero."""
    plant = Plant(rig, seed=0)
    joints = "".join(
        f'<joint joint1="hinge{i + 1}" polycoef="0 0 0 0 0" solref="0.001 1"/>'
        for i in range(rig.n_links)
    )
    xml = make_rig_xml(rig).replace("</mujoco>", f"<equality>{joints}</equality></mujoco>")
    plant.model = mujoco.MjModel.from_xml_string(xml)
    plant.data = mujoco.MjData(plant.model)
    plant.reset()
    return plant


def test_reset_absolute_angles_and_world_link_points():
    plant = Plant(RigParams())
    theta = np.array([0.1, -0.2, 0.3, -0.4])
    plant.reset(theta, cart_x=0.12)
    np.testing.assert_allclose(plant.state()["q"], absolute_to_relative(theta))
    np.testing.assert_allclose(plant.state()["theta"], theta)
    np.testing.assert_array_equal(plant.data.qvel, 0)
    points = plant.link_points()
    assert points.shape == (5, 2)
    np.testing.assert_allclose(points[0], [0.12, 0.14])
    lengths = np.array([link.length for link in plant.rig.links])
    np.testing.assert_allclose(
        np.diff(points, axis=0), lengths[:, None] * np.column_stack((np.sin(theta), np.cos(theta)))
    )
    # Returned diagnostics must not expose writable views into the simulator.
    state = plant.state()
    state["q"][:] = 9
    state["theta"][:] = 9
    state["omega"][:] = 9
    np.testing.assert_allclose(plant.data.qpos[1:], absolute_to_relative(theta))
    np.testing.assert_array_equal(plant.data.qvel, 0)


def test_step_advances_exact_control_period_and_reset_clears_drive_and_push():
    plant = Plant(RigParams())
    plant.push(3, 2, duration=0.1)
    for _ in range(7):
        plant.step(5)
    assert plant.t == pytest.approx(7 * plant.rig.control_dt, abs=1e-14)
    assert plant.state()["v_setpoint"] == pytest.approx(0.035)
    plant.reset()
    assert plant.t == 0
    assert not plant.hit_rail_end
    for key in ("x", "v", "force", "v_setpoint"):
        assert plant.state()[key] == 0
    np.testing.assert_array_equal(plant.data.ctrl, 0)
    np.testing.assert_array_equal(plant.data.xfrc_applied, 0)
    plant.step(0)
    np.testing.assert_allclose(plant.data.qpos, 0, atol=1e-15)


def test_velocity_servo_tracks_acceleration_with_links_held_upright():
    plant = held_upright_plant(RigParams())
    tail = []
    for i in range(200):
        plant.step(5.0)
        state = plant.state()
        assert abs(state["force"]) <= plant.rig.drive.force_limit
        if i >= 150:
            tail.append(state["a"])
    assert plant.t == pytest.approx(0.2)
    assert plant.state()["v_setpoint"] == pytest.approx(1.0)
    assert plant.state()["v"] == pytest.approx(1.0, abs=0.015)
    np.testing.assert_allclose(tail, 5.0, rtol=0.03)
    assert np.max(np.abs(plant.state()["theta"])) < 0.001


def test_drive_limits_and_anti_windup_recover_after_saturation():
    base = RigParams()
    rig = replace(
        base,
        cart=replace(base.cart, friction=0, viscous=0, rail_half_length=10),
        drive=replace(base.drive, force_limit=2.0, accel_limit=5.0, velocity_limit=0.4),
    )
    plant = held_upright_plant(rig)
    plant.step(1e6)
    assert plant.state()["v_setpoint"] == pytest.approx(5 * rig.control_dt)
    forces = []
    for _ in range(199):
        plant.step(1e6)
        forces.append(plant.state()["force"])
        assert abs(plant.data.actuator_force[0]) <= 2.0
        assert abs(plant.state()["v_setpoint"]) <= 0.4
    assert max(forces) == pytest.approx(2.0)
    # Reversing the ramp takes the setpoint below actual velocity. A wound-up
    # integral would continue commanding a positive force after that crossing.
    for _ in range(100):
        plant.step(-1e6)
        state = plant.state()
        if state["v_setpoint"] < state["v"] - 0.01:
            assert state["force"] < 0
            break
    else:
        pytest.fail("The velocity setpoint never crossed below feedback")


def test_measurement_quantization_and_relative_encoder_convention():
    base = RigParams()
    rig = replace(base, sensors=replace(base.sensors, encoder_noise=0, cart_noise=0))
    plant = Plant(rig)
    quantum = 2 * np.pi / 2**rig.sensors.encoder_bits
    q_counts = np.array([2.2, -3.3, 7.49, -1.51])
    plant.reset(np.cumsum(q_counts * quantum), cart_x=12.7 * rig.sensors.cart_resolution)
    measurement = plant.measure()
    assert measurement.t == plant.t
    assert measurement.cart_position == pytest.approx(13 * rig.sensors.cart_resolution)
    np.testing.assert_allclose(measurement.joint_angles / quantum, [2, -3, 7, -2])


def test_measurement_noise_is_seeded_and_precedes_quantization():
    base = RigParams()
    rig = replace(base, sensors=replace(base.sensors, cart_noise=3e-5, encoder_noise=0.001))
    plant = Plant(rig, seed=41)
    other = Plant(rig, seed=41)
    oracle = np.random.default_rng(41)
    quantum = 2 * np.pi / 2**rig.sensors.encoder_bits
    for _ in range(4):
        measurement = plant.measure()
        matching = other.measure()
        expected_cart = oracle.normal(0, rig.sensors.cart_noise)
        expected_q = oracle.normal(0, rig.sensors.encoder_noise, rig.n_links)
        assert measurement.cart_position == pytest.approx(
            np.rint(expected_cart / rig.sensors.cart_resolution) * rig.sensors.cart_resolution
        )
        np.testing.assert_allclose(
            measurement.joint_angles, np.rint(expected_q / quantum) * quantum
        )
        assert measurement.cart_position == matching.cart_position
        np.testing.assert_array_equal(measurement.joint_angles, matching.joint_angles)


def test_push_is_a_force_at_the_moving_tip_and_expires():
    plant = Plant(RigParams())
    theta = np.array([0.1, -0.05, 0.2, 0.3])
    plant.reset(theta)
    plant.push(3, 2.0, duration=0.0013)
    body_id = plant.model.body("link4").id
    lever = plant.rig.links[3].length - plant.rig.links[3].com
    np.testing.assert_allclose(
        plant.data.xfrc_applied[body_id], [2, 0, 0, 0, 2 * lever * np.cos(theta[3]), 0]
    )
    plant.step(0)
    assert plant.data.xfrc_applied[body_id, 0] == 2.0
    plant.step(0)
    np.testing.assert_array_equal(plant.data.xfrc_applied, 0)


def test_positive_push_gives_positive_tip_velocity():
    plant = Plant(RigParams())
    plant.push(3, 1.0)
    for _ in range(20):
        plant.step(0)
    state = plant.state()
    lengths = np.array([link.length for link in plant.rig.links])
    tip_vx = state["v"] + np.sum(lengths * np.cos(state["theta"]) * state["omega"])
    assert tip_vx > 0.05
    assert plant.link_points()[-1, 0] > 0
    np.testing.assert_array_equal(plant.data.xfrc_applied, 0)


def test_uncontrolled_tiny_tilt_falls_within_three_seconds():
    plant = Plant(RigParams())
    plant.reset(np.full(plant.rig.n_links, 0.002))
    for _ in range(3000):
        plant.step(0)
        if plant.fallen():
            break
    assert plant.fallen()
    assert np.max(np.abs(plant.state()["theta"])) > 0.6
    assert plant.t < 3


def test_rail_end_flag_is_latched_until_reset():
    plant = Plant(RigParams())
    plant.reset(cart_x=plant.rig.cart.rail_half_length - 1e-5)
    plant.data.qvel[0] = 0.2
    plant.step(0)
    assert plant.hit_rail_end
    assert plant.fallen()
    plant.data.qpos[0] = 0
    assert plant.fallen()
    plant.reset()
    assert not plant.hit_rail_end
    assert not plant.fallen()


def test_invalid_substep_timing_rejected():
    rig = RigParams()
    with pytest.raises(ValueError, match="evenly divide"):
        Plant(replace(rig, timing=replace(rig.timing, physics_dt=0.0003)))


def test_fallen_is_measured_from_the_configuration_equilibrium():
    plant = Plant(RigParams())
    theta = equilibrium_angles("UUUD", 4)
    plant.reset(theta)
    # Default target is upright, so a hanging top link counts as fallen there.
    assert plant.fallen()
    assert not plant.fallen(theta_eq=theta)
    plant.reset(theta + 0.1)
    assert not plant.fallen(theta_eq=theta)
    assert plant.fallen(theta_eq=theta, limit_rad=0.05)
