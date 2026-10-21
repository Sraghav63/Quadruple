"""MuJoCo's physical model of the measured cart and serial pendulum rig."""

from __future__ import annotations

from quad_pendulum.params import RigParams


def make_rig_xml(rig: RigParams) -> str:
    """Build a rig with relative encoder joints and force input in newtons."""
    if rig.n_links < 1:
        raise ValueError("The rig needs at least one link")
    stop = rig.cart.rail_half_length + 0.05
    height = rig.total_height
    colors = ("0.88 0.30 0.22 1", "0.12 0.62 0.48 1", "0.96 0.70 0.20 1", "0.51 0.35 0.78 1")
    bodies = []
    for i, link in enumerate(rig.links):
        z = 0.07 if i == 0 else rig.links[i - 1].length
        color = colors[i % len(colors)]
        bodies.append(f"""
      <body name="link{i + 1}" pos="0 0 {z}">
        <joint name="hinge{i + 1}" type="hinge" axis="0 1 0"
          frictionloss="{rig.joints.coulomb}" damping="{rig.joints.viscous}" armature="0"/>
        <inertial pos="0 0 {link.com}" mass="{link.mass}"
          diaginertia="{link.inertia_com} {link.inertia_com} {link.inertia_com * 0.001}"/>
        <geom name="link{i + 1}" type="capsule" fromto="0 0 0 0 0 {link.length}"
          size="0.006" rgba="{color}"/>
        <geom type="sphere" pos="0 0 {link.length}" size="0.018" rgba="{color}"/>
        <site name="tip{i + 1}" pos="0 0 {link.length}" size="0.003" rgba="{color}"/>
""")
    chain = "".join(bodies) + "</body>\n" * rig.n_links
    # Implicit velocity integration handles damping and Coulomb/stiction constraints
    # at every physics step; RK4 does not offer fourth-order constraint integration.
    return f"""<mujoco model="servo_cart_{rig.n_links}_pendulum">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{rig.timing.physics_dt}" integrator="implicitfast"
    gravity="0 0 {-rig.gravity}" solver="Newton" tolerance="1e-12" iterations="100"/>
  <default>
    <geom contype="0" conaffinity="0" density="0"/>
  </default>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
      rgb1="0.85 0.87 0.89" rgb2="0.72 0.75 0.79"/>
    <material name="floor" texture="grid" texrepeat="8 4"/>
  </asset>
  <worldbody>
    <light pos="-2 -3 4" dir="0.4 0.6 -1" diffuse="0.9 0.88 0.84"/>
    <light pos="2 1 3" dir="-0.4 -0.2 -1" diffuse="0.5 0.6 0.7"/>
    <camera name="side" pos="0 {-max(2.5, height * 2.6)} {0.2 + height / 2}"
      xyaxes="1 0 0 0 0 1" fovy="40"/>
    <geom name="floor" type="plane" pos="0 0 -0.05" size="3 2 0.1" material="floor"/>
    <geom name="rail" type="box" pos="0 0 0" size="{stop + 0.10} 0.055 0.018"
      rgba="0.16 0.19 0.23 1"/>
    <geom name="left_stop" type="box" pos="{-stop - 0.12} 0 0.05"
      size="0.025 0.1 0.065" rgba="0.7 0.18 0.16 1"/>
    <geom name="right_stop" type="box" pos="{stop + 0.12} 0 0.05"
      size="0.025 0.1 0.065" rgba="0.7 0.18 0.16 1"/>
    <body name="cart" pos="0 0 0.07">
      <joint name="slider" type="slide" axis="1 0 0" limited="true"
        range="{-stop} {stop}" frictionloss="{rig.cart.friction}"
        damping="{rig.cart.viscous}" armature="0"/>
      <inertial pos="0 0 0" mass="{rig.cart.mass}"
        diaginertia="{rig.cart.mass * (0.08**2 + 0.04**2) / 3}
                     {rig.cart.mass * (0.12**2 + 0.04**2) / 3}
                     {rig.cart.mass * (0.12**2 + 0.08**2) / 3}"/>
      <geom name="cart" type="box" size="0.12 0.08 0.04" rgba="0.1 0.3 0.55 1"/>
      <geom type="sphere" pos="0 0 0.07" size="0.022" rgba="0.7 0.73 0.76 1"/>
      {chain}
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_force" joint="slider" gear="1" ctrllimited="true"
      ctrlrange="{-rig.drive.force_limit} {rig.drive.force_limit}"/>
  </actuator>
</mujoco>
"""
