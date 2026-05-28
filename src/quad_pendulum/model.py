"""MuJoCo XML generation for cart-pendulum models."""

from __future__ import annotations


def make_cart_pendulum_xml(
    links: int = 4,
    *,
    link_length: float = 0.45,
    link_radius: float = 0.025,
    cart_size: tuple[float, float, float] = (0.18, 0.12, 0.08),
    timestep: float = 0.004,
) -> str:
    """Create a MuJoCo XML model for a cart with 1 to 4 inverted pendulum links."""
    if not 1 <= links <= 4:
        raise ValueError("links must be between 1 and 4")

    cart_x, cart_y, cart_z = cart_size
    link_bodies = _make_link_bodies(
        links=links,
        link_length=link_length,
        link_radius=link_radius,
        attach_z=cart_z + 0.02,
    )

    return f"""<mujoco model="cart_{links}_link_pendulum">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="RK4" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.25 0.25 0.25" specular="0.15 0.15 0.15"/>
    <rgba haze="0.78 0.86 0.94 1"/>
    <global azimuth="90" elevation="-18" offwidth="1280" offheight="720"/>
  </visual>

  <default>
    <joint damping="0.004" armature="0.001" frictionloss="0.001"/>
    <geom friction="0.7 0.005 0.0001" density="500"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
      rgb1="0.86 0.88 0.86" rgb2="0.74 0.77 0.74"/>
    <material name="floor_grid" texture="grid" texrepeat="10 3" reflectance="0.12"/>
    <material name="rail_dark" rgba="0.08 0.09 0.10 1" specular="0.35" shininess="0.4"/>
    <material name="cart_blue" rgba="0.08 0.28 0.52 1" specular="0.45" shininess="0.5"/>
    <material name="rubber" rgba="0.03 0.03 0.035 1"/>
    <material name="metal" rgba="0.72 0.74 0.76 1" specular="0.55" shininess="0.7"/>
  </asset>

  <worldbody>
    <camera name="side" pos="0 -5.4 1.65" xyaxes="1 0 0 0 0.22 0.98" fovy="34"/>
    <light pos="-2.5 -3 4" dir="0.4 0.7 -1" diffuse="0.9 0.88 0.82"/>
    <light pos="2.5 -2 2.5" dir="-0.6 0.4 -1" diffuse="0.35 0.42 0.5"/>
    <geom name="floor" type="plane" pos="0 0 -0.08" size="4.2 1.15 0.1" material="floor_grid"
      contype="0" conaffinity="0" density="0"/>
    <geom name="rear_rail" type="box" pos="0 0.13 0" size="3.2 0.018 0.018" material="rail_dark"
      contype="0" conaffinity="0" density="0"/>
    <geom name="front_rail" type="box" pos="0 -0.13 0" size="3.2 0.018 0.018" material="rail_dark"
      contype="0" conaffinity="0" density="0"/>
    <geom name="left_stop" type="box" pos="-3.05 0 0.08" size="0.035 0.23 0.13" rgba="0.55 0.13 0.12 1"
      contype="0" conaffinity="0" density="0"/>
    <geom name="right_stop" type="box" pos="3.05 0 0.08" size="0.035 0.23 0.13" rgba="0.55 0.13 0.12 1"
      contype="0" conaffinity="0" density="0"/>
    <geom name="center_mark" type="box" pos="0 0 -0.055" size="0.01 0.28 0.006" rgba="0.08 0.09 0.1 1" contype="0" conaffinity="0" density="0"/>
    <geom name="left_limit_mark" type="box" pos="-2.4 0 -0.055" size="0.01 0.24 0.006" rgba="0.65 0.12 0.10 1" contype="0" conaffinity="0" density="0"/>
    <geom name="right_limit_mark" type="box" pos="2.4 0 -0.055" size="0.01 0.24 0.006" rgba="0.65 0.12 0.10 1" contype="0" conaffinity="0" density="0"/>

    <body name="cart" pos="0 0 {cart_z}">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-3 3"/>
      <geom name="cart" type="box" size="{cart_x} {cart_y} {cart_z}" material="cart_blue"/>
      <geom name="cart_top" type="box" pos="0 0 {cart_z + 0.018}" size="{cart_x * 0.72} {cart_y * 0.78} 0.018"
        rgba="0.93 0.96 0.98 1" contype="0" conaffinity="0" density="0"/>
      <geom name="front_left_wheel" type="cylinder" fromto="-0.11 -0.145 -0.055 -0.11 -0.105 -0.055"
        size="0.045" material="rubber" contype="0" conaffinity="0" density="0"/>
      <geom name="front_right_wheel" type="cylinder" fromto="0.11 -0.145 -0.055 0.11 -0.105 -0.055"
        size="0.045" material="rubber" contype="0" conaffinity="0" density="0"/>
      <geom name="rear_left_wheel" type="cylinder" fromto="-0.11 0.105 -0.055 -0.11 0.145 -0.055"
        size="0.045" material="rubber" contype="0" conaffinity="0" density="0"/>
      <geom name="rear_right_wheel" type="cylinder" fromto="0.11 0.105 -0.055 0.11 0.145 -0.055"
        size="0.045" material="rubber" contype="0" conaffinity="0" density="0"/>
      <geom name="hinge_block" type="sphere" pos="0 0 {cart_z + 0.02}" size="0.052"
        material="metal" contype="0" conaffinity="0" density="0"/>
      {link_bodies}
    </body>
  </worldbody>

  <actuator>
    <motor name="cart_force" joint="slider" gear="1" ctrlrange="-100 100" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def _make_link_bodies(
    *,
    links: int,
    link_length: float,
    link_radius: float,
    attach_z: float,
) -> str:
    indent = "      "
    lines: list[str] = []

    for index in range(1, links + 1):
        name = f"link{index}"
        pos_z = attach_z if index == 1 else link_length
        rgba = _link_color(index)
        lines.append(
            f'{indent}<body name="{name}" pos="0 0 {pos_z}">\n'
            f'{indent}  <joint name="hinge{index}" type="hinge" axis="0 1 0"/>\n'
            f'{indent}  <geom name="{name}" type="capsule" fromto="0 0 0 0 0 {link_length}" '
            f'size="{link_radius}" rgba="{rgba}"/>\n'
            f'{indent}  <geom name="{name}_tip" type="sphere" pos="0 0 {link_length}" '
            f'size="{link_radius * 1.6}" rgba="{rgba}"/>\n'
        )
        indent += "  "

    for index in range(links, 0, -1):
        indent = "      " + "  " * (index - 1)
        lines.append(f"{indent}</body>\n")

    return "".join(lines).rstrip()


def _link_color(index: int) -> str:
    colors = {
        1: "0.82 0.28 0.20 1",
        2: "0.14 0.55 0.34 1",
        3: "0.92 0.68 0.18 1",
        4: "0.45 0.28 0.68 1",
    }
    return colors[index]
