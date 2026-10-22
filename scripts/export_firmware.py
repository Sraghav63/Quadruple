#!/usr/bin/env python3
"""Export the nominal controller, parameter provenance and firmware limits."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from quad_pendulum.controller import LQRController
from quad_pendulum.params import RigParams, parse_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "firmware/teensy/quad_pendulum/qp_gains.h",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="UUUD",
        help="links balanced up (U) or down (D), bottom link first; default all up",
    )
    args = parser.parse_args()
    rig = RigParams()
    try:
        configuration = parse_configuration(args.config, rig.n_links)
    except ValueError as exc:
        parser.error(str(exc))
    controller = LQRController(rig, configuration=configuration)
    # The upstream formatter emits e.g. 0f / 30f. C++ requires a decimal point
    # or exponent before f; repair lexical spelling without changing any value.
    header = re.sub(r"(?<![\w.+-])(-?\d+)f\b", r"\1.0f", controller.to_c_header())
    provenance = (
        "/* RigParams used to design these gains (SI units):\n"
        + json.dumps(asdict(rig), indent=2)
        + "\n*/\n"
    )
    limits = (
        f"#define QP_RAIL_LIMIT {rig.cart.rail_half_length:.9e}f\n"
        f"#define QP_VELOCITY_LIMIT {rig.drive.velocity_limit:.9e}f\n"
        f"#define QP_CONTROL_US {round(rig.control_dt * 1e6)}\n"
        f"#define QP_COMPUTE_DELAY_STEPS {rig.timing.compute_delay_steps}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(provenance + header + limits)
    print(
        f"Wrote {args.output} (configuration={configuration}, NZ={controller.nz}, NY={1 + rig.n_links}, "
        f"closed-loop radius={controller.closed_loop_radius:.8f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
