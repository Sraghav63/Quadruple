import numpy as np

from quad_pendulum import QuadPendulumEnv


def make_deterministic_env(**kwargs) -> QuadPendulumEnv:
    defaults = {
        "links": 1,
        "control_latency_steps": 0,
        "actuator_time_constant": 0.0,
        "max_force_rate": float("inf"),
        "control_deadband": 0.0,
        "force_noise_std": 0.0,
        "disturbance_force_std": 0.0,
        "observation_noise_scale": 0.0,
    }
    defaults.update(kwargs)
    return QuadPendulumEnv(**defaults)


def test_control_latency_delays_force_application():
    env = make_deterministic_env(control_latency_steps=1)
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.asarray([1.0], dtype=np.float32))
    assert info["target_force"] == 80.0
    assert info["delayed_target_force"] == 0.0
    assert info["applied_force"] == 0.0

    _, _, _, _, info = env.step(np.asarray([1.0], dtype=np.float32))
    assert info["delayed_target_force"] == 80.0
    assert info["applied_force"] == 80.0
    env.close()


def test_force_rate_limit_caps_motor_force_change():
    env = make_deterministic_env(max_force_rate=10.0)
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.asarray([1.0], dtype=np.float32))

    assert np.isclose(info["applied_force"], 0.2)
    env.close()


def test_observation_noise_changes_observed_state_without_changing_sim_state():
    env = QuadPendulumEnv(
        links=1,
        force_noise_std=0.0,
        disturbance_force_std=0.0,
        observation_noise_scale=10.0,
    )
    env.reset(seed=0)
    qpos_before = env.data.qpos.copy()

    first_obs = env._get_obs()
    second_obs = env._get_obs()

    assert not np.allclose(first_obs, second_obs)
    assert np.allclose(env.data.qpos, qpos_before)
    env.close()
