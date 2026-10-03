"""Command-line runner for the polynomial-defined mirror."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from AFM_module import define_mirror, test_mirror

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Define and test a synchrotron free-form mirror")
    parser.add_argument("config", type=Path, nargs="?")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())
    # The focal geometry is specified at the first mirror point. The angle is
    # measured from the direction toward the first beam point to the focal
    # point, so an angle of 90 degrees places the focus directly above it.
    focal_angle = math.radians(float(config["focal_angle_degrees"]))
    focal_distance = float(config["focal_distance"])
    mirror_distance = float(config["mirror_distance"])
    focus_x = mirror_distance - focal_distance * math.cos(focal_angle)
    focus_y = focal_distance * math.sin(focal_angle)

    print(f"{CYAN}Free-form mirror simulation{RESET}")
    print(f"Configuration: {config_path}")
    started = time.perf_counter()
    polynomial, domain = define_mirror(
        beam_points=int(config["defining_beam_points"]),
        bending_radius=float(config["bending_radius"]),
        bending_angle=float(config["bending_angle"]),
        mirror_distance=float(config["mirror_distance"]),
        focus_x=focus_x,
        focus_y=focus_y,
        polynomial_degree=int(config["polynomial_degree"]),
        show_progress=True,
    )
    define_seconds = time.perf_counter() - started
    print(f"{GREEN}Mirror definition complete{RESET} ({define_seconds:.2f}s)")
    print(f"Mirror: {polynomial}")
    print(f"Domain: {domain}")
    started = time.perf_counter()
    result = test_mirror(
        beam_points=int(config["testing_beam_points"]),
        bending_radius=float(config["bending_radius"]),
        bending_angle=float(config["bending_angle"]),
        focus_x=focus_x,
        focus_y=focus_y,
        mirror_polynomial_latex=polynomial,
        mirror_domain=domain,
        rays_per_beam_point=int(config.get("rays_per_beam_point", 25)),
        angular_distribution_points=config["angular_distribution_points"],
        angular_distribution_initial_guess=config.get(
            "angular_distribution_initial_guess",
            [9000.0, 3059.47, 2344.43],
        ),
        seed=int(config.get("seed", 20260906)),
        collection_fractions=config.get("collection_fractions", []),
        mirror_segments=int(config.get("mirror_segments", 1000)),
        show_model=bool(config.get("show_model", False)),
        intensity_sharpness=float(config.get("intensity_sharpness", 0)),
        angular_spread_sharpness=float(config.get("angular_spread_sharpness", 0)),
        show_progress=True,
    )
    test_seconds = time.perf_counter() - started
    print(f"{GREEN}Mirror testing complete{RESET} ({test_seconds:.2f}s)")
    print(f"Hits: {result['hit_count']:,}/{result['ray_count']:,}")
    print(f"{YELLOW}Mirror function (LaTeX):{RESET} {polynomial}")
    print(f"{YELLOW}Mirror domain:{RESET} {domain}")
    if "peak_intensity" in result:
        print(f"{YELLOW}Peak intensity:{RESET} {result['peak_intensity']:.6g}")
    for fraction, distance in result.get("collection_distances_micrometers", {}).items():
        print(
            f"{YELLOW}{100 * fraction:g}% ray collection width:{RESET} "
            f"{distance:.6g} µm"
        )
    print(f"Total runtime: {define_seconds + test_seconds:.2f}s")
    if config.get("show_model", False) or (
        float(config.get("intensity_sharpness", 0)) > 0
        or float(config.get("angular_spread_sharpness", 0)) > 0
    ):
        import matplotlib.pyplot as plt
        print("Close the graph windows to finish.")
        plt.show()


if __name__ == "__main__":
    main()
