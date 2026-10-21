"""Physical description of the rig, shared by the plant, the controller and the tools.

Every number here should eventually be replaced by a measurement of the real
build. The defaults describe a buildable rig: four 0.35 m links of 12 mm steel
rod, a bearing/encoder housing at each upper joint, a belt-driven cart on a
1.2 m MGN12 rail with a brushless servo in velocity mode, 14-bit magnetic
encoders and a 1 kHz control loop.

Conventions used across the package:
- Joint angles q_i are what the encoders read: link i relative to link i-1
  (link 0 is the cart), zero when straight up, positive tilting toward +x.
- Absolute angles theta_i = q_1 + ... + q_i are each link's tilt from vertical.
- The controller's input is a cart acceleration command in m/s^2. The servo
  drive turns it into a velocity setpoint and closes a velocity loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class LinkParams:
    """One pendulum link, modelled as a uniform rod plus a point mass at its top."""

    length: float  # hinge-to-hinge, m
    rod_mass: float  # kg, spread uniformly along the length
    top_mass: float  # kg at the upper end: next joint's bearings, encoder, magnet, screws

    @property
    def mass(self) -> float:
        return self.rod_mass + self.top_mass

    @property
    def com(self) -> float:
        """Distance of the centre of mass from the lower hinge, m."""
        return (self.rod_mass * self.length / 2 + self.top_mass * self.length) / self.mass

    @property
    def inertia_com(self) -> float:
        """Moment of inertia about the centre of mass (hinge axis direction), kg m^2."""
        c = self.com
        rod = self.rod_mass * (self.length**2 / 12 + (self.length / 2 - c) ** 2)
        top = self.top_mass * (self.length - c) ** 2
        return rod + top


@dataclass(frozen=True)
class JointParams:
    """Bearing losses at each pendulum hinge (relative motion between links)."""

    coulomb: float = 5e-4  # N m, dry/stiction torque per joint (two small shielded bearings)
    viscous: float = 2e-5  # N m s/rad


@dataclass(frozen=True)
class CartParams:
    mass: float = 1.2  # kg: carriage, plate, joint-1 housing
    rail_half_length: float = 0.6  # m of usable travel either side of centre
    friction: float = 1.5  # N, rail + belt Coulomb friction (felt by the drive)
    viscous: float = 2.0  # N s/m


@dataclass(frozen=True)
class DriveParams:
    """Servo drive in velocity mode: accel command -> velocity setpoint -> force."""

    accel_limit: float = 30.0  # m/s^2, clamp on the commanded acceleration
    velocity_limit: float = 2.5  # m/s, clamp on the velocity setpoint
    force_limit: float = 150.0  # N at the belt (motor torque limit / pulley radius)
    velocity_bandwidth: float = 200.0  # rad/s, closed velocity-loop bandwidth
    # The effective first-order lag from accel command to cart acceleration that
    # the controller designs against. Roughly 1 / velocity_bandwidth.
    model_lag: float = 0.005  # s


@dataclass(frozen=True)
class SensorParams:
    encoder_bits: int = 14  # AS5047P-class absolute magnetic encoder per joint
    encoder_noise: float = 2e-4  # rad RMS electrical noise before quantisation
    cart_resolution: float = 1e-5  # m per count of the cart position sensor
    cart_noise: float = 0.0  # m RMS


@dataclass(frozen=True)
class TimingParams:
    control_rate: float = 1000.0  # Hz, controller loop on the microcontroller
    physics_dt: float = 0.0002  # s, plant integration step (must divide the control period)
    compute_delay_steps: int = 1  # control periods between sampling and the command taking effect


@dataclass(frozen=True)
class RigParams:
    links: tuple[LinkParams, ...] = field(
        default_factory=lambda: tuple(
            [LinkParams(0.35, 0.31, 0.06)] * 3 + [LinkParams(0.35, 0.31, 0.0)]
        )
    )
    joints: JointParams = field(default_factory=JointParams)
    cart: CartParams = field(default_factory=CartParams)
    drive: DriveParams = field(default_factory=DriveParams)
    sensors: SensorParams = field(default_factory=SensorParams)
    timing: TimingParams = field(default_factory=TimingParams)
    gravity: float = GRAVITY

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def control_dt(self) -> float:
        return 1.0 / self.timing.control_rate

    @property
    def physics_substeps(self) -> int:
        steps = self.control_dt / self.timing.physics_dt
        if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
            raise ValueError("physics_dt must evenly divide the control period")
        return round(steps)

    @property
    def total_height(self) -> float:
        return float(sum(link.length for link in self.links))

    def with_links(self, lengths, rod_masses, top_mass: float = 0.06) -> RigParams:
        """Convenience for design sweeps: rebuild the chain with new geometry."""
        n = len(lengths)
        links = tuple(
            LinkParams(float(lengths[i]), float(rod_masses[i]), top_mass if i < n - 1 else 0.0)
            for i in range(n)
        )
        return replace(self, links=links)


def relative_to_absolute(q: np.ndarray) -> np.ndarray:
    """Encoder joint angles (or rates) -> absolute link angles (or rates)."""
    return np.cumsum(q, axis=-1)


def absolute_to_relative(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return np.diff(theta, axis=-1, prepend=0.0)


@dataclass
class Measurement:
    """What the microcontroller reads each control period. All values quantised."""

    t: float  # s, sample time
    cart_position: float  # m
    joint_angles: np.ndarray  # rad, relative encoder angles q_i, shape (n_links,)
