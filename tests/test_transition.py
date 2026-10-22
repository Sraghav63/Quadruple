"""Real-time transitions between up/down equilibria: planned knots, run-time design,
the route library, and the closed loop through the simulation supervisor."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from quad_pendulum.controller import LQRController
from quad_pendulum.linear import linearized
from quad_pendulum.params import (
    Measurement,
    absolute_to_relative,
    equilibrium_angles,
    wrap_angle,
)
from quad_pendulum.presets import preset_rig
from quad_pendulum.sim import Simulation
from quad_pendulum.transition import (
    DATA_DIR,
    PlanLimits,
    Trajectory,
    TransitionController,
    TransitionLibrary,
    chain_jacobian,
    dare_doubling,
    plan_transition,
)


def _library_file(name: str) -> str:
    return str(DATA_DIR / "links2" / name)


# ----------------------------------------------------------------- 1. JSON
def test_trajectory_json_round_trip(tmp_path):
    lib = TransitionLibrary(DATA_DIR / "links1")
    traj = lib.get("D", "U")
    path = tmp_path / "D-U.json"
    traj.save(path)
    loaded = Trajectory.load(path)
    assert (loaded.start, loaded.goal) == ("D", "U")
    assert loaded.duration == pytest.approx(traj.duration)
    assert loaded.limits == traj.limits
    for name in ("x", "v", "a", "jerk", "theta", "omega", "alpha"):
        a, b = getattr(traj, name), getattr(loaded, name)
        np.testing.assert_array_equal(a, b, err_msg=name)
        assert a.shape == b.shape


# ------------------------------------------------------------- 2. sampling
@pytest.mark.parametrize("name", ["DD-UU.json"])
def test_sample_matches_knots_and_shapes(name):
    rig = preset_rig("links2")
    traj = Trajectory.load(_library_file(name))
    knots = traj.x.size
    h = traj.duration / (knots - 1)
    dt = h / 5  # every knot falls on a control sample
    lag = rig.drive.model_lag
    t, state, u = traj.sample(dt, lag)
    n = traj.n_links
    steps = round(traj.duration / dt)

    assert t.shape == (steps + 1,)
    assert state.shape == (3 + 2 * n, steps + 1)
    assert u.shape == (steps + 1,)

    idx = np.arange(knots) * 5
    np.testing.assert_allclose(state[0, idx], traj.x, atol=1e-9)
    np.testing.assert_allclose(state[1, idx], traj.v, atol=1e-9)
    np.testing.assert_allclose(state[2, idx], traj.a, atol=1e-9)
    np.testing.assert_allclose(state[3 : 3 + n, idx], traj.theta, atol=1e-9)
    np.testing.assert_allclose(state[3 + n :, idx], traj.omega, atol=1e-9)

    # Last sample is the last knot.
    np.testing.assert_allclose(state[0, -1], traj.x[-1], atol=1e-9)
    np.testing.assert_allclose(state[3 : 3 + n, -1], traj.theta[:, -1], atol=1e-9)
    np.testing.assert_allclose(state[3 + n :, -1], traj.omega[:, -1], atol=1e-9)

    # Commanded acceleration is a + model_lag * jerk.
    np.testing.assert_allclose(u[idx], traj.a + lag * traj.jerk, atol=1e-9)


# ------------------------------------------------------------ 3. Jacobian
@pytest.mark.parametrize("cfg", ["UU", "DU", "UD", "DD"])
def test_chain_jacobian_matches_linearized_at_equilibria(cfg):
    rig = preset_rig("links2")
    n = rig.n_links
    theta_eq = equilibrium_angles(cfg, n)
    jac = chain_jacobian(rig, theta_eq, np.zeros(n), 0.0)
    k, d, bv = linearized(rig, cfg)
    assert jac.shape == (n, 2 * n + 1)
    np.testing.assert_allclose(jac[:, 0], bv.reshape(-1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(jac[:, 1 : 1 + n], k, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(jac[:, 1 + n :], d, rtol=1e-5, atol=1e-6)


def test_chain_jacobian_batch_equals_single_calls():
    rig = preset_rig("links2")
    rng = np.random.default_rng(7)
    k = 5
    theta = rng.uniform(-0.4, 0.4, (k, 2)) + equilibrium_angles("DU", 2)
    omega = rng.uniform(-1.0, 1.0, (k, 2))
    cart_accel = rng.uniform(-3.0, 3.0, k)
    batch = chain_jacobian(rig, theta, omega, cart_accel)
    assert batch.shape == (k, 2, 5)
    for i in range(k):
        single = chain_jacobian(rig, theta[i], omega[i], cart_accel[i])
        np.testing.assert_allclose(batch[i], single, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ 4. DARE
def test_dare_doubling_matches_scipy():
    rig = preset_rig("links2")
    ctrl = LQRController(rig, configuration="UU")
    a, b = ctrl.az, ctrl.bz
    q = np.eye(a.shape[0])
    r = np.array([[1.0]])
    p_ref = solve_discrete_are(a, b, q, r)
    p = dare_doubling(a, b, q, r)
    assert p.shape == p_ref.shape
    np.testing.assert_allclose(p, p_ref, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------- 5. routing
def _fake(start: str, goal: str, duration: float) -> Trajectory:
    z = np.zeros(3)
    m = np.zeros((1, 3))
    return Trajectory(
        start=start,
        goal=goal,
        duration=duration,
        x=z,
        v=z,
        a=z,
        jerk=z,
        theta=m,
        omega=m,
        alpha=m,
    )


def test_route_prefers_quicker_two_hop_over_longer_direct():
    lib = TransitionLibrary(
        trajectories=[
            _fake("DD", "UU", 5.0),  # direct, slow
            _fake("DD", "DU", 1.0),
            _fake("DU", "UU", 1.0),  # two hops, 2.0 s total
        ]
    )
    assert lib.route("DD", "UU") == [("DD", "DU"), ("DU", "UU")]


def test_route_direct_edge_and_empty_and_unreachable():
    lib = TransitionLibrary(
        trajectories=[_fake("DD", "DU", 1.0), _fake("DU", "UU", 1.0)]
    )
    assert lib.route("DD", "DU") == [("DD", "DU")]
    assert lib.route("DD", "DD") == []
    assert lib.route("UU", "DD") is None  # no edges leave UU
    assert lib.route("DD", "UD") is None  # UD never appears


def test_route_on_planned_library_is_connected():
    lib = TransitionLibrary(DATA_DIR / "links2")
    for start in ("DD", "DU", "UD", "UU"):
        for goal in ("DD", "DU", "UD", "UU"):
            route = lib.route(start, goal)
            assert route is not None, (start, goal)
            if route:
                assert route[0][0] == start and route[-1][1] == goal


# ------------------------------------------------------ 6. TransitionController
def _measurement(t: float, cart: float, theta_abs: np.ndarray) -> Measurement:
    return Measurement(t=t, cart_position=float(cart), joint_angles=absolute_to_relative(theta_abs))


def test_transition_controller_tracks_nominal_exactly():
    rig = preset_rig("links2")
    lib = TransitionLibrary(DATA_DIR / "links2")
    traj = lib.get("DD", "UU")
    tc = TransitionController(rig, traj)
    n = rig.n_links
    ns = 3 + 2 * n
    assert tc.gains.shape == (tc.steps, tc.nz)
    assert tc.nz == ns + rig.timing.compute_delay_steps

    tc.begin(np.zeros(ns), np.zeros(rig.timing.compute_delay_steps))
    # Exact nominal measurements: the commands must reproduce u_issue throughout.
    for k in range(tc.steps):
        assert not tc.done
        meas = _measurement(tc.t[k], tc.nominal[0, k], tc.nominal[3 : 3 + n, k])
        cmd = tc.step(meas)
        assert abs(cmd - tc.u_issue[k]) < 1e-6, (k, cmd, tc.u_issue[k])
    assert tc.done
    assert tc.progress == pytest.approx(1.0)


def test_transition_controller_done_after_steps_calls():
    rig = preset_rig("links2")
    traj = TransitionLibrary(DATA_DIR / "links2").get("DD", "UU")
    tc = TransitionController(rig, traj)
    n = rig.n_links
    tc.begin(np.zeros(3 + 2 * n), np.zeros(rig.timing.compute_delay_steps))
    for k in range(tc.steps):
        assert not tc.done
        tc.step(_measurement(tc.t[k], tc.nominal[0, k], tc.nominal[3 : 3 + n, k]))
    assert tc.k == tc.steps
    assert tc.done


def test_transition_controller_x_offset_shifts_frame():
    rig = preset_rig("links2")
    traj = TransitionLibrary(DATA_DIR / "links2").get("DD", "UU")
    tc = TransitionController(rig, traj)
    n = rig.n_links
    ns = 3 + 2 * n
    delay = rig.timing.compute_delay_steps
    offset = 0.03

    def run(x_offset: float, shift: float) -> np.ndarray:
        dev = np.zeros(ns)
        dev[0] = x_offset  # physical cart position relative to the centre, as the sim passes it
        tc.begin(dev, np.zeros(delay), x_offset=x_offset)
        cmds = []
        for k in range(200):
            cmds.append(
                tc.step(_measurement(tc.t[k], tc.nominal[0, k] + shift, tc.nominal[3 : 3 + n, k]))
            )
        return np.array(cmds)

    reference = run(0.0, 0.0)
    shifted = run(offset, offset)
    np.testing.assert_allclose(shifted, reference, atol=1e-6)


# ------------------------------------------------------ 7. closed loop (sim)
def _advance_until(sim: Simulation, predicate, limit_s: float) -> set[str]:
    """Step until `predicate()`; returns the set of modes visited on the way."""
    modes = {sim.mode}
    for _ in range(round(limit_s * sim.rig.timing.control_rate)):
        sim.step()
        modes.add(sim.mode)
        assert not sim.fallen, f"fell at t={sim.t:.3f}s: {sim.plant.state()}"
        if predicate():
            return modes
    raise AssertionError(f"not reached within {limit_s:.2f} s; status={sim.status()}")


def test_links1_down_up_down_closed_loop():
    rig = preset_rig("links1")
    lib = TransitionLibrary(DATA_DIR / "links1")
    sim = Simulation(rig, seed=11, configuration="D", transitions=lib)
    sim.reset(random_tilt_deg=0.3)
    sim.run(0.3)
    assert not sim.fallen

    for goal in ("U", "D"):
        assert sim.request_configuration(goal) is True
        duration = max(lib.get(*edge).duration for edge in lib.edges)
        modes = _advance_until(sim, lambda: sim.mode == "hold", duration + 3.0)
        assert "transition" in modes, "the request did not run a planned transition"
        assert sim.configuration == goal
        assert not sim.fallen
    sim.run(0.5)
    assert not sim.fallen


def test_links2_multi_hop_closed_loop():
    rig = preset_rig("links2")
    lib = TransitionLibrary(DATA_DIR / "links2")
    sim = Simulation(rig, seed=12, configuration="DD", transitions=lib)
    sim.reset(random_tilt_deg=0.3)
    sim.run(0.3)
    assert not sim.fallen

    for goal in ("UU", "DU"):
        route = lib.route(sim.configuration, goal)
        assert route is not None
        assert sim.request_configuration(goal) is True
        budget = sum(lib.get(*edge).duration for edge in route) + 3.0 * len(route) + 3.0
        modes = _advance_until(sim, lambda: sim.mode == "hold", budget)
        assert "transition" in modes
        assert sim.configuration == goal
        assert not sim.fallen


def test_request_configuration_without_library_is_false():
    sim = Simulation(preset_rig("links1"), seed=13, configuration="D")
    assert sim.request_configuration("U") is False
    assert sim.mode == "hold"


def test_request_current_configuration_is_hold():
    sim = Simulation(preset_rig("links1"), seed=14, configuration="D", transitions=TransitionLibrary(DATA_DIR / "links1"))
    sim.reset(random_tilt_deg=0.0)
    assert sim.request_configuration("D") is True
    assert sim.route == []
    assert sim.mode == "hold"


# ------------------------------------------------------------ 8. planning
def test_plan_transition_smoke_links1_down_to_up():
    pytest.importorskip("casadi")
    rig = preset_rig("links1")
    limits = PlanLimits()
    # Seed 0 does not converge at this grid (the random guess is the only difference;
    # the planner script retries over seeds for that reason), so the smoke test pins seed 1.
    traj = plan_transition(rig, "D", "U", 2.0, knots=101, coarse_knots=51, seed=1)
    assert traj is not None, "IPOPT did not converge"
    assert isinstance(traj, Trajectory)
    assert (traj.start, traj.goal) == ("D", "U")
    assert traj.theta.shape == (1, 101)
    assert traj.x.shape == (101,)
    # Ends upright, at rest, with the cart centred (mod whole turns).
    assert abs(wrap_angle(traj.theta[0, -1])) < 1e-3
    assert abs(traj.omega[0, -1]) < 1e-3
    assert abs(traj.x[-1]) < 1e-3
    # Starts hanging, at rest.
    assert wrap_angle(traj.theta[0, 0] - np.pi) == pytest.approx(0.0, abs=1e-6)
    assert np.abs(traj.x).max() <= limits.rail + 1e-6
