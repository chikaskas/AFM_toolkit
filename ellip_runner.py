"""Compare the configured beam against an elliptical focusing mirror.

This script intentionally leaves the free-form mirror module, runner, and
configuration file unchanged. It uses the same configuration values and
reports the same screen metrics as the normal runner.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from AFM_module import (
    _angular_profile,
    _beam_points,
    _central_collection_distances,
    _fit_symmetric_gaussians,
    _screen_profile,
)

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

AXIS_FONT_SIZE = 16
TICK_FONT_SIZE = 14

def _ellipse_from_config(config: dict) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return foci, ellipse axes, and rotation from the configured geometry."""
    mirror_distance = float(config["mirror_distance"])
    focal_angle = math.radians(float(config["focal_angle_degrees"]))
    focal_distance = float(config["focal_distance"])
    focus_2 = np.array(
        (
            mirror_distance - focal_distance * math.cos(focal_angle),
            focal_distance * math.sin(focal_angle),
        ),
        dtype=float,
    )
    focus_1 = np.zeros(2)
    first_mirror_point = np.array((mirror_distance, 0.0))
    perimeter = np.linalg.norm(first_mirror_point - focus_1) + np.linalg.norm(
        first_mirror_point - focus_2
    )
    semi_major = perimeter / 2.0
    focal_half_distance = np.linalg.norm(focus_2 - focus_1) / 2.0
    if semi_major <= focal_half_distance:
        raise ValueError("The configured focal geometry cannot form an ellipse")
    semi_minor = math.sqrt(semi_major**2 - focal_half_distance**2)
    rotation = math.atan2(focus_2[1], focus_2[0])
    return focus_1, focus_2, semi_major, semi_minor, rotation


def _sample_rays(
    config: dict, focus_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    beam_points = int(config["testing_beam_points"])
    rays_per_point = int(config["rays_per_beam_point"])
    x_beam, y_beam, tangent = _beam_points(
        float(config["bending_radius"]),
        float(config["bending_angle"]),
        beam_points,
    )
    _, peak, sigma = _fit_symmetric_gaussians(
        config["angular_distribution_points"],
        config.get("angular_distribution_initial_guess", [9000.0, 3059.47, 2344.43]),
    )
    seed = int(config.get("seed", 20260906))
    rng = np.random.default_rng(None if seed == 0 else seed)
    peak_sign = np.where(
        rng.integers(0, 2, size=(beam_points, rays_per_point)) == 0,
        -peak,
        peak,
    )
    offsets_microrad = peak_sign + rng.normal(
        0.0, sigma, size=(beam_points, rays_per_point)
    )
    rays = np.repeat(
        np.column_stack((x_beam, y_beam, tangent)),
        rays_per_point,
        axis=0,
    )
    rays[:, 2] += offsets_microrad.ravel() * 1e-6
    return rays, offsets_microrad.ravel(), float(sigma)


def _trace_ellipse(
    rays: np.ndarray,
    focus_1: np.ndarray,
    focus_2: np.ndarray,
    semi_major: float,
    semi_minor: float,
    rotation: float,
    first_mirror_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intersect rays with the ellipse and reflect them using its normal."""
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    center = (focus_1 + focus_2) / 2.0
    direction_x = np.cos(rays[:, 2])
    direction_y = np.sin(rays[:, 2])
    relative_x = rays[:, 0] - center[0]
    relative_y = rays[:, 1] - center[1]
    local_origin_x = cos_r * relative_x + sin_r * relative_y
    local_origin_y = -sin_r * relative_x + cos_r * relative_y
    local_direction_x = cos_r * direction_x + sin_r * direction_y
    local_direction_y = -sin_r * direction_x + cos_r * direction_y
    qa = (local_direction_x / semi_major) ** 2 + (
        local_direction_y / semi_minor
    ) ** 2
    qb = 2.0 * (
        local_origin_x * local_direction_x / semi_major**2
        + local_origin_y * local_direction_y / semi_minor**2
    )
    qc = (local_origin_x / semi_major) ** 2 + (
        local_origin_y / semi_minor
    ) ** 2 - 1.0
    discriminant = qb * qb - 4.0 * qa * qc
    hit = discriminant >= 0.0
    hit_x = np.full(len(rays), np.nan)
    hit_y = np.full(len(rays), np.nan)
    reflected = np.full((len(rays), 2), np.nan)
    valid_indices = np.flatnonzero(hit)
    if len(valid_indices):
        root_distance = np.sqrt(np.maximum(discriminant[valid_indices], 0.0))
        roots = np.column_stack(
            (
                (-qb[valid_indices] - root_distance) / (2.0 * qa[valid_indices]),
                (-qb[valid_indices] + root_distance) / (2.0 * qa[valid_indices]),
            )
        )
        roots[roots <= 1e-10] = np.inf
        root_points = np.stack(
            (
                rays[valid_indices, 0, None] + roots * direction_x[valid_indices, None],
                rays[valid_indices, 1, None] + roots * direction_y[valid_indices, None],
            ),
            axis=2,
        )
        root_error = np.linalg.norm(root_points - first_mirror_point, axis=2)
        root_error[~np.isfinite(roots)] = np.inf
        selected_root = np.argmin(root_error, axis=1)
        distance = roots[np.arange(len(selected_root)), selected_root]
        valid = np.isfinite(distance)
        selected = valid_indices[valid]
        distance = distance[valid]
        hit_x[selected] = rays[selected, 0] + distance * direction_x[selected]
        hit_y[selected] = rays[selected, 1] + distance * direction_y[selected]
        hit[selected] = True
        p = np.column_stack((hit_x[selected], hit_y[selected]))
        normal = (p - focus_1) / np.linalg.norm(p - focus_1, axis=1)[:, None]
        normal += (p - focus_2) / np.linalg.norm(p - focus_2, axis=1)[:, None]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        incoming = np.column_stack((direction_x[selected], direction_y[selected]))
        reflected[selected] = incoming - 2.0 * np.sum(incoming * normal, axis=1)[:, None] * normal
    return hit, hit_x, hit_y, reflected


def main() -> None:
    parser = argparse.ArgumentParser(description="Test an elliptical focusing mirror")
    parser.add_argument("config", type=Path, nargs="?")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())

    focus_1, focus_2, semi_major, semi_minor, rotation = _ellipse_from_config(config)
    first_mirror_point = np.array((float(config["mirror_distance"]), 0.0))
    rays, offsets, sigma = _sample_rays(config, focus_2)
    started = time.perf_counter()
    hit, hit_x, hit_y, reflected = _trace_ellipse(
        rays,
        focus_1,
        focus_2,
        semi_major,
        semi_minor,
        rotation,
        first_mirror_point,
    )
    trace_seconds = time.perf_counter() - started
    fractions = [float(value) for value in config.get("collection_fractions", [])]
    widths_meters = _central_collection_distances(
        hit, hit_x, hit_y, reflected, tuple(focus_2), fractions
    )
    widths = {
        fraction: distance * 1e6
        for fraction, distance in widths_meters.items()
    }
    intensity_sharpness = float(config.get("intensity_sharpness", 0.0))
    angular_sharpness = float(config.get("angular_spread_sharpness", 0.0))
    intensity_x = intensity_y = angular_x = angular_y = None
    if intensity_sharpness > 0:
        intensity_x, intensity_y, _ = _screen_profile(
            hit, hit_x, hit_y, reflected, focus_2, intensity_sharpness
        )

    print(f"{CYAN}Elliptical mirror test{RESET}")
    print(f"Configuration: {config_path}")
    print(f"Focus 1: (0.0, 0.0)")
    print(f"Focus 2: ({focus_2[0]:.9g}, {focus_2[1]:.9g})")
    print(f"Ellipse axes: a={semi_major:.9g} m, b={semi_minor:.9g} m")
    print(f"{GREEN}Ray tracing complete{RESET} ({trace_seconds:.2f}s)")
    print(f"Hits: {int(hit.sum()):,}/{len(rays):,}")
    for fraction, width in widths.items():
        print(f"{YELLOW}{100 * fraction:g}% ray collection width:{RESET} {width:.6g} µm")
    if intensity_y is not None:
        print(f"{YELLOW}Peak intensity:{RESET} {float(intensity_y.max()):.6g}")

    if angular_sharpness > 0:
        angular_x, angular_y = _angular_profile(
            offsets,
            sigma,
            angular_sharpness,
        )

    if config.get("show_model", False):
        fig, axis = plt.subplots(figsize=(11, 7))
        parameter = np.linspace(0.0, 2.0 * math.pi, 2000)
        local = np.column_stack((semi_major * np.cos(parameter), semi_minor * np.sin(parameter)))
        rotation_matrix = np.array(((math.cos(rotation), -math.sin(rotation)),
                                    (math.sin(rotation), math.cos(rotation))))
        ellipse = local @ rotation_matrix.T + (focus_1 + focus_2) / 2.0
        # Plot only the physical branch struck by the rays. Parameterizing
        # from the actual hit range avoids an empty mask and never displays
        # the unrelated far side of the mathematical ellipse.
        valid_hit = hit & np.isfinite(hit_x)
        relative_hit = np.column_stack((hit_x[valid_hit], hit_y[valid_hit])) - (focus_1 + focus_2) / 2.0
        local_hit_x = rotation_matrix.T[0, 0] * relative_hit[:, 0] + rotation_matrix.T[0, 1] * relative_hit[:, 1]
        local_hit_y = rotation_matrix.T[1, 0] * relative_hit[:, 0] + rotation_matrix.T[1, 1] * relative_hit[:, 1]
        hit_parameter = np.unwrap(np.arctan2(local_hit_y / semi_minor, local_hit_x / semi_major))
        branch = np.linspace(hit_parameter.min(), hit_parameter.max(), 500)
        branch_local = np.column_stack((semi_major * np.cos(branch), semi_minor * np.sin(branch)))
        branch_world = branch_local @ rotation_matrix.T + (focus_1 + focus_2) / 2.0
        axis.plot(branch_world[:, 0], branch_world[:, 1], "r", label="Mirror")
        all_x = np.concatenate((rays[:, 0], hit_x[valid_hit], [focus_1[0], focus_2[0]]))
        all_y = np.concatenate((rays[:, 1], hit_y[valid_hit], [focus_1[1], focus_2[1]]))
        axis.set_xlim(float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)
        axis.set_ylim(float(all_y.min()) - 1.0, float(all_y.max()) + 1.0)
        arc = np.linspace(0.0, float(config["bending_angle"]), 400)
        axis.plot(
            float(config["bending_radius"]) * np.sin(arc),
            float(config["bending_radius"]) * (np.cos(arc) - 1.0),
            "k--",
            label="Beam arc",
        )
        # Sample across the complete beam-point range, rather than taking the
        # first flattened rays. This keeps every part of the electron arc
        # visible even when many rays per point are used.
        draw = np.linspace(0, len(rays) - 1, min(len(rays), 3000), dtype=int)
        ray_count = max(1, len(draw))
        ray_alpha = float(np.clip(18.0 / np.sqrt(ray_count), 0.015, 0.16))
        ray_width = float(np.clip(1.2 / np.sqrt(ray_count / 100.0), 0.12, 0.45))
        drawn = draw[hit[draw]]
        incident_segments = np.stack(
            (
                rays[drawn, :2],
                np.column_stack((hit_x[drawn], hit_y[drawn])),
            ),
            axis=1,
        )
        axis.add_collection(LineCollection(
            incident_segments, colors="tab:blue", alpha=ray_alpha, linewidths=ray_width
        ))
        reflected_start = np.column_stack((hit_x[drawn], hit_y[drawn]))
        reflected_segments = np.stack(
            (reflected_start, reflected_start + 30.0 * reflected[drawn]), axis=1
        )
        axis.add_collection(LineCollection(
            reflected_segments, colors="tab:orange", alpha=ray_alpha, linewidths=ray_width
        ))
        direction = np.nanmean(reflected[hit], axis=0)
        direction /= np.linalg.norm(direction)
        screen_normal = np.array((-direction[1], direction[0]))
        screen_half_length = max(1.0, 0.08 * semi_major)
        screen = np.vstack(
            (focus_2 - screen_half_length * screen_normal,
             focus_2 + screen_half_length * screen_normal)
        )
        axis.plot(screen[:, 0], screen[:, 1], color="black", linewidth=1.2, label="Screen")
        axis.plot(*focus_1, "ko", markersize=5, label="Focus 1")
        axis.plot(*focus_2, "ro", markersize=5, label="Focus 2")
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel("y (m)", fontsize=AXIS_FONT_SIZE)
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        axis.grid(True, linestyle=":")
        axis.legend()
        plt.show(block=False)
        plt.pause(0.001)

    if intensity_x is not None:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(intensity_x, intensity_y, color="purple")
        axis.set_xlabel("Screen displacement (micrometres)", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel("Relative intensity density", fontsize=AXIS_FONT_SIZE)
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        axis.grid(True, linestyle=":")
    if angular_x is not None:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(angular_x, angular_y, color="tab:blue", linewidth=2.5)
        axis.fill_between(angular_x, angular_y, color="tab:blue", alpha=0.2)
        axis.set_xlabel("Angular offset (microradians)", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel("Frequency density", fontsize=AXIS_FONT_SIZE)
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        axis.grid(True, linestyle=":")
    if config.get("show_model", False) or intensity_x is not None or angular_x is not None:
        plt.show(block=False)
        plt.pause(0.001)
        print("Close the graph windows to finish.")
        plt.show()


if __name__ == "__main__":
    main()
