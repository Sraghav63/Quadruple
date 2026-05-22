"""Quadruple inverted pendulum RL environment."""

__all__ = ["QuadPendulumEnv"]


def __getattr__(name: str):
    if name == "QuadPendulumEnv":
        from quad_pendulum.env import QuadPendulumEnv

        return QuadPendulumEnv
    raise AttributeError(name)
