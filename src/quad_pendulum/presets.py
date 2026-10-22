"""Named rigs shared by the web export and the transition planner."""

from __future__ import annotations

from quad_pendulum.params import RigParams


def presets() -> list[tuple[str, str, str, RigParams]]:
    quad = RigParams()
    link = quad.links[0]
    out = [
        ("quad", "Quadruple (4 links)", "The build target: 4 x 0.35 m, 12 mm steel rod.", quad),
    ]
    for n, name in [(3, "Triple"), (2, "Double"), (1, "Single")]:
        rig = quad.with_links([link.length] * n, [link.rod_mass] * n, top_mass=link.top_mass)
        out.append((f"links{n}", f"{name} ({n} link{'s' if n > 1 else ''})", "Same links, fewer of them.", rig))
    short = quad.with_links([0.30] * 4, [0.27] * 4, top_mass=link.top_mass)
    out.append(
        ("quad-short", "Quadruple, 0.30 m links", "Shorter, faster, less robust (85% vs ~97% in Monte Carlo).", short)
    )
    return out


def preset_rig(preset_id: str) -> RigParams:
    for pid, _, _, rig in presets():
        if pid == preset_id:
            return rig
    raise KeyError(f"unknown preset {preset_id!r}")
