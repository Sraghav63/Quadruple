"""Behavioral checks against the nonlinear, quantised MuJoCo closed loop."""

import numpy as np

from quad_pendulum.sim import Simulation


def advance(sim: Simulation, seconds: float) -> None:
    for _ in range(round(seconds / sim.rig.control_dt)):
        sim.step()
        assert not sim.fallen, f"Fell at {sim.t:.3f} s: {sim.plant.state()}"


def test_nominal_stays_upright_five_seconds():
    sim = Simulation(seed=0)
    sim.reset(theta=np.full(4, np.deg2rad(0.05)))
    advance(sim, 5.0)


def test_survives_light_taps_on_every_link():
    # A quadruple pendulum only tolerates light taps: ~0.1 N for 20 ms on the top
    # link (2 mN s) is near the limit with a 30 m/s^2 cart.
    for link, force in enumerate([0.3, 0.2, 0.1, 0.08]):
        sim = Simulation(seed=1)
        sim.reset(random_tilt_deg=0.05)
        advance(sim, 1.0)
        sim.push(link, force, 0.020)
        advance(sim, 3.0)


def test_tracks_twenty_centimetre_reference_step():
    sim = Simulation(seed=2)
    sim.reset(random_tilt_deg=0.05)
    advance(sim, 1.0)
    sim.set_reference(0.2)
    # Joint stiction makes the cart hunt around the target, so judge the average.
    xs = []
    for _ in range(round(6.0 / sim.rig.control_dt)):
        sim.step()
        assert not sim.fallen
        xs.append(sim.plant.state()["x"])
    assert abs(np.mean(xs[-3000:]) - sim.x_ref) < 0.08


def test_falls_with_controller_disabled():
    sim = Simulation(seed=3)
    sim.reset(theta=np.full(4, np.deg2rad(0.05)))
    sim.controller_enabled = False
    for _ in range(round(5.0 / sim.rig.control_dt)):
        sim.step()
        if sim.fallen:
            break
    assert sim.fallen, "An unbalanced upright chain should fall without active control"
