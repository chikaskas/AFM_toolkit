"""Free-form synchrotron mirror definition and testing.

The coordinate convention follows the report and the supplied reference code:
the electron beam starts at (0, 0), initially travels in +x, and bends toward
negative y.  All distances are metres and all angles passed to the public API
are radians except ray offsets, which are microradians.
"""

from __future__ import annotations

import math
import re
import time
from typing import Sequence

import numba
import numpy as np
from matplotlib.collections import LineCollection

__all__ = ["define_mirror", "test_mirror"]
AXIS_FONT_SIZE = 16
TICK_FONT_SIZE = 14


def _beam_points(radius: float, bend_angle: float, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.linspace(0.0, bend_angle, count)
    return (
        radius * np.sin(alpha),
        radius * (np.cos(alpha) - 1.0),
        -alpha,
    )


def _line_intersection(p: np.ndarray, t: np.ndarray, q: np.ndarray, u: np.ndarray) -> tuple[float, float] | None:
    cross = t[0] * u[1] - t[1] * u[0]
    if abs(cross) < 1e-14:
        return None
    delta = q - p
    s = (delta[0] * u[1] - delta[1] * u[0]) / cross
    ray_distance = (delta[0] * t[1] - delta[1] * t[0]) / cross
    return s, ray_distance


def _latex_polynomial(coefficients: np.ndarray, variable: str = "x") -> str:
    def format_number(value: float) -> str:
        if value == 0:
            return "0"
        text = f"{abs(value):.12g}"
        if "e" not in text.lower():
            return text
        mantissa, exponent = text.lower().split("e")
        exponent_value = int(exponent)
        return rf"{mantissa}*10^{{{exponent_value}}}"

    terms: list[str] = []
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        if abs(coefficient) < 1e-14:
            continue
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(float(coefficient))
        number = format_number(magnitude)
        if power == 0:
            body = number
        elif power == 1:
            body = f"{number}{variable}"
        else:
            body = f"{number}{variable}^{{{power}}}"
        terms.append((sign, body))
    if not terms:
        return r"$y=0$"
    first_sign, first_body = terms[0]
    result = ("" if first_sign == "+" else "-") + first_body
    for sign, body in terms[1:]:
        result += f" {sign} {body}"
    return rf"$y={result}$"


@numba.njit
def _define_mirror_kernel(
    focus_x: float,
    focus_y: float,
    x_beam: np.ndarray,
    y_beam: np.ndarray,
    beam_directions: np.ndarray,
    mirror: np.ndarray,
    start_idx: int,
    end_idx: int,
):
    """Calculates a chunk of the mirror surface."""
    for i in range(start_idx, end_idx):
        incoming_x = beam_directions[i, 0]
        incoming_y = beam_directions[i, 1]

        outgoing_x = focus_x - mirror[i, 0]
        outgoing_y = focus_y - mirror[i, 1]
        norm_out = np.sqrt(outgoing_x * outgoing_x + outgoing_y * outgoing_y)
        outgoing_x /= norm_out
        outgoing_y /= norm_out

        tangent_x = incoming_x + outgoing_x
        tangent_y = incoming_y + outgoing_y
        tangent_norm = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
        if tangent_norm < 1e-13:
            raise ValueError("Degenerate reflection bisector at beam point")
        tangent_x /= tangent_norm
        tangent_y /= tangent_norm

        u0 = beam_directions[i + 1, 0]
        u1 = beam_directions[i + 1, 1]
        cross = tangent_x * u1 - tangent_y * u0
        if abs(cross) < 1e-14:
            raise ValueError("Parallel rays encountered during mirror definition")

        delta0 = x_beam[i + 1] - mirror[i, 0]
        delta1 = y_beam[i + 1] - mirror[i, 1]
        s = (delta0 * u1 - delta1 * u0) / cross
        ray_dist = (delta0 * tangent_y - delta1 * tangent_x) / cross

        if ray_dist <= 0:
            raise ValueError("Could not construct mirror point")

        mirror[i + 1, 0] = mirror[i, 0] + s * tangent_x
        mirror[i + 1, 1] = mirror[i, 1] + s * tangent_y
def define_mirror(
    beam_points: int,
    bending_radius: float,
    bending_angle: float,
    mirror_distance: float,
    focus_x: float,
    focus_y: float,
    polynomial_degree: int,
    show_progress: bool = False,
) -> tuple[str, tuple[float, float]]:
    
    if beam_points < 2:
        raise ValueError("beam_points must be at least 2")
    if bending_radius <= 0 or mirror_distance <= 0:
        raise ValueError("bending_radius and mirror_distance must be positive")
    if bending_angle <= 0:
        raise ValueError("bending_angle must be positive")
    if polynomial_degree < 1 or polynomial_degree >= beam_points:
        raise ValueError("polynomial_degree must be in [1, beam_points - 1]")

    x_beam, y_beam, ray_angles = _beam_points(bending_radius, bending_angle, beam_points)
    beam_directions = np.column_stack((np.cos(ray_angles), np.sin(ray_angles)))

    mirror = np.empty((beam_points, 2), dtype=float)
    mirror[0] = (mirror_distance, 0.0)
    
    total_steps = beam_points - 1
    chunk_size = max(1, total_steps // 100)
    started = time.perf_counter()

    for start_idx in range(0, total_steps, chunk_size):
        end_idx = min(start_idx + chunk_size, total_steps)
        
        _define_mirror_kernel(
            float(focus_x), float(focus_y), x_beam, y_beam, beam_directions, mirror, start_idx, end_idx
        )
        
        if show_progress:
            elapsed = time.perf_counter() - started
            rate = end_idx / max(elapsed, 1e-9)
            remaining = (total_steps - end_idx) / max(rate, 1e-9)
            print(
                f"\r\033[2KDefine {100 * end_idx / total_steps:5.1f}% "
                f"ETA {remaining:5.1f}s",
                end="",
                flush=True,
            )
            
    if show_progress:
        print("\r\033[2K", end="")

    coefficients = np.polynomial.Polynomial.fit(
        mirror[:, 0], mirror[:, 1], polynomial_degree
    ).convert().coef
    coefficients = coefficients[::-1]
    
    return _latex_polynomial(coefficients), (float(mirror[:, 0].min()), float(mirror[:, 0].max()))

def _trace_polynomial(
    rays: np.ndarray,
    coefficients: np.ndarray,
    domain: tuple[float, float],
    show_progress: bool = False,
    progress_offset: int = 0,
    progress_total: int | None = None,
    progress_started: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n = len(rays)
    hit = np.zeros(n, dtype=bool)
    hit_x = np.full(n, np.nan)
    hit_y = np.full(n, np.nan)
    reflected = np.full((n, 2), np.nan)

    started = progress_started if progress_started is not None else time.perf_counter()
    total_work = progress_total or n
    batch_size = 50000
    
    derivative = np.polyder(coefficients)
    grid_size = 300
    x_grid = np.linspace(domain[0], domain[1], grid_size)
    p_grid = np.polyval(coefficients, x_grid)

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        
        x0 = rays[start_idx:end_idx, 0]
        y0 = rays[start_idx:end_idx, 1]
        vx = np.cos(rays[start_idx:end_idx, 2])
        vy = np.sin(rays[start_idx:end_idx, 2])

        valid_rays = np.abs(vx) >= 1e-14
        if not np.any(valid_rays):
            continue

        m = vy[valid_rays] / vx[valid_rays]
        x0_v = x0[valid_rays]
        y0_v = y0[valid_rays]
        vx_v = vx[valid_rays]

        g_grid = p_grid[None, :] - (y0_v[:, None] + m[:, None] * (x_grid[None, :] - x0_v[:, None]))
        t_grid = (x_grid[None, :] - x0_v[:, None]) / vx_v[:, None]

        valid_t = (t_grid[:, :-1] > 1e-10) & (t_grid[:, 1:] > 1e-10)
        sign_change = (g_grid[:, :-1] * g_grid[:, 1:] <= 0) & valid_t

        has_hit = np.any(sign_change, axis=1)
        if not np.any(has_hit):
            continue

        first_hit_idx = np.argmax(sign_change[has_hit], axis=1)
        valid_indices = np.flatnonzero(valid_rays)[has_hit]
        
        x_left = x_grid[first_hit_idx]
        x_right = x_grid[first_hit_idx + 1]
        x_curr = 0.5 * (x_left + x_right)

        m_sub = m[has_hit]
        x0_sub = x0_v[has_hit]
        y0_sub = y0_v[has_hit]
        vx_sub = vx_v[has_hit]

        for _ in range(8):
            p_val = np.polyval(coefficients, x_curr)
            dp_val = np.polyval(derivative, x_curr)
            g_val = p_val - (y0_sub + m_sub * (x_curr - x0_sub))
            dg_val = dp_val - m_sub
            x_curr = x_curr - g_val / dg_val

        t_hit = (x_curr - x0_sub) / vx_sub
        hit_mask = (t_hit > 1e-10) & (x_curr >= domain[0]) & (x_curr <= domain[1])

        final_valid = valid_indices[hit_mask]
        x_final = x_curr[hit_mask]

        global_valid = start_idx + final_valid
        
        hit[global_valid] = True
        hit_x[global_valid] = x_final
        hit_y[global_valid] = y0[final_valid] + (x_final - x0[final_valid]) * (vy[final_valid] / vx[final_valid])

        slope = np.polyval(derivative, x_final)
        norm = np.sqrt(1.0 + slope * slope)
        nx, ny = -slope / norm, 1.0 / norm
        dot = vx[final_valid] * nx + vy[final_valid] * ny
        
        reflected[global_valid, 0] = vx[final_valid] - 2.0 * dot * nx
        reflected[global_valid, 1] = vy[final_valid] - 2.0 * dot * ny

        completed = progress_offset + end_idx
        if show_progress:
            elapsed = time.perf_counter() - started
            rate = completed / max(elapsed, 1e-9)
            remaining = (total_work - completed) / max(rate, 1e-9)
            print(
                f"\r\033[2KTrace {100 * completed / total_work:5.1f}% "
                f"ETA {remaining:5.1f}s",
                end="",
                flush=True,
            )

    return hit, hit_x, hit_y, reflected

def _projected_ray_domain(
    rays: np.ndarray,
    coefficients: np.ndarray,
    domain: tuple[float, float],
) -> tuple[float, float]:
    """Estimate the required x-domain from the two extreme input rays."""
    candidate_rays = np.array((rays[np.argmin(rays[:, 2])], rays[np.argmax(rays[:, 2])]))
    projected_x: list[float] = []
    for x0, y0, angle in candidate_rays:
        dx, dy = math.cos(angle), math.sin(angle)
        if abs(dx) < 1e-14:
            continue
        equation = coefficients.copy()
        slope = dy / dx
        equation[-1] -= y0 - x0 * slope
        equation[-2] -= slope
        roots = np.roots(equation)
        real = roots.real[np.abs(roots.imag) < 1e-8]
        forward = (real - x0) / dx > 1e-10
        real = real[forward]
        if real.size:
            # A high-degree polynomial can have distant forward roots. Select
            # the root associated with the supplied mirror domain, not the
            # first algebraic root encountered along the ray.
            distance_from_domain = np.maximum(
                domain[0] - real,
                np.maximum(real - domain[1], 0.0),
            )
            projected_x.append(float(real[np.argmin(distance_from_domain)]))
    if len(projected_x) != 2:
        return domain
    lower, upper = sorted(projected_x)
    width = max(upper - lower, domain[1] - domain[0])
    margin = 1e-9 * width
    return lower - margin, upper + margin

def _screen_profile(
    hit: np.ndarray,
    hit_x: np.ndarray,
    hit_y: np.ndarray,
    reflected: np.ndarray,
    focus: tuple[float, float],
    sharpness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    valid = hit & np.isfinite(reflected[:, 0])
    fx, fy = focus
    direction = np.nanmean(reflected[valid], axis=0)
    direction /= np.linalg.norm(direction)
    normal = np.array((-direction[1], direction[0]))
    delta = np.column_stack((hit_x[valid] - fx, hit_y[valid] - fy))
    denominator = delta @ direction
    # The screen normal is perpendicular to the average reflected ray.
    screen_parameter = -(delta @ direction) / (reflected[valid] @ direction)
    screen_points = np.column_stack((hit_x[valid], hit_y[valid])) + screen_parameter[:, None] * reflected[valid]
    displacement_um = (screen_points - np.array(focus)) @ normal * 1e6
    bins = int(round(1000 + 1000 * sharpness))
    grid = np.linspace(displacement_um.min(), displacement_um.max(), bins)
    bandwidth = 0.35 / sharpness * np.std(displacement_um) * (
        4.0 / (3.0 * len(displacement_um))
    ) ** 0.2
    bandwidth = max(float(bandwidth), 1e-9)
    # Histogram first, then smooth the fixed-size grid. This avoids the
    # O(number_of_rays * bins) memory and runtime cost of a full KDE matrix.
    counts, edges = np.histogram(
        displacement_um, bins=bins, range=(grid[0], grid[-1]), density=False
    )
    spacing = max(float(grid[1] - grid[0]), 1e-12)
    sigma_bins = max(float(bandwidth / spacing), 0.5)
    radius = min(
        max(1, int(math.ceil(4.0 * sigma_bins))),
        max(1, (len(counts) - 1) // 2),
    )
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(counts.astype(float), kernel, mode="same")
    density /= len(displacement_um) * spacing
    return grid, density, float(density.max())

def _angular_profile(
    offsets_microrad: np.ndarray,
    sigma_microrad: float,
    sharpness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the configured smoothed angular density profile."""
    spread_min = float(offsets_microrad.min())
    spread_max = float(offsets_microrad.max())
    spread_range = max(spread_max - spread_min, 1.0)
    spread_grid = np.linspace(
        spread_min - 0.08 * spread_range,
        spread_max + 0.08 * spread_range,
        int(round(1000 + 1000 * sharpness)),
    )
    bandwidth = max(
        0.35 / sharpness * float(sigma_microrad),
        spread_range / 100.0,
    )
    counts, edges = np.histogram(
        offsets_microrad,
        bins=len(spread_grid) - 1,
        range=(spread_grid[0], spread_grid[-1]),
    )
    spacing = max(float(edges[1] - edges[0]), 1e-12)
    sigma_bins = max(float(bandwidth / spacing), 0.5)
    from scipy.ndimage import gaussian_filter1d

    density = gaussian_filter1d(
        counts.astype(float),
        sigma=sigma_bins,
        mode="constant",
    )
    density /= len(offsets_microrad) * spacing
    return 0.5 * (edges[:-1] + edges[1:]), density


def _fit_symmetric_gaussians(
    distribution_points: Sequence[Sequence[float]],
    initial_guess: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """Fit two equal, symmetric Gaussian peaks to [offset, density] points."""
    from scipy.optimize import curve_fit

    points = np.asarray(distribution_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(
            "angular_distribution_points must contain at least three [x, y] pairs"
        )
    if np.all(points[:, 0] <= 0.0) or np.all(points[:, 0] >= 0.0):
        mirrored = points[np.abs(points[:, 0]) > 0.0].copy()
        mirrored[:, 0] *= -1.0
        points = np.vstack((points, mirrored))

    def model(x: np.ndarray, amplitude: float, peak: float, sigma: float) -> np.ndarray:
        return amplitude * (
            np.exp(-((x - peak) ** 2) / (2.0 * sigma**2))
            + np.exp(-((x + peak) ** 2) / (2.0 * sigma**2))
        )

    if initial_guess is None:
        initial_guess = [9000.0, 3059.47, 2344.43]
    if len(initial_guess) != 3:
        raise ValueError(
            "angular_distribution_initial_guess must contain [A, B, C]"
        )
    fitted, _ = curve_fit(
        model,
        points[:, 0],
        points[:, 1],
        p0=np.asarray(initial_guess, dtype=float),
        maxfev=10000,
    )
    return tuple(map(float, fitted))


def _central_collection_distances(
    hit: np.ndarray,
    hit_x: np.ndarray,
    hit_y: np.ndarray,
    reflected: np.ndarray,
    focus: tuple[float, float],
    fractions: Sequence[float],
) -> dict[float, float]:
    """Return the shortest centered screen interval for each ray fraction."""
    valid = hit & np.isfinite(reflected[:, 0])
    direction = np.nanmean(reflected[valid], axis=0)
    direction /= np.linalg.norm(direction)
    normal = np.array((-direction[1], direction[0]))
    delta = np.column_stack((hit_x[valid] - focus[0], hit_y[valid] - focus[1]))
    screen_parameter = -(delta @ direction) / (reflected[valid] @ direction)
    screen_points = np.column_stack((hit_x[valid], hit_y[valid]))
    screen_points += screen_parameter[:, None] * reflected[valid]
    displacement = (screen_points - np.asarray(focus)) @ normal
    result: dict[float, float] = {}
    for fraction in fractions:
        lower = 50.0 * (1.0 - fraction)
        upper = 100.0 - lower
        result[float(fraction)] = float(
            np.percentile(displacement, upper) - np.percentile(displacement, lower)
        )
    return result


def test_mirror(
    beam_points: int,
    bending_radius: float,
    bending_angle: float,
    focus_x: float,
    focus_y: float,
    mirror_polynomial_latex: str,
    mirror_domain: tuple[float, float],
    rays_per_beam_point: int = 25,
    angular_distribution_points: Sequence[Sequence[float]] = (),
    angular_distribution_initial_guess: Sequence[float] | None = None,
    seed: int = 20260906,
    collection_fractions: Sequence[float] = (),
    mirror_segments: int = 1000,
    show_model: bool = False,
    intensity_sharpness: float = 0.0,
    angular_spread_sharpness: float = 0.0,
    max_domain_expansions: int = 8,
    show_progress: bool = False,
) -> dict[str, object]:
    """Test a mirror with vectorized bimodal-Gaussian angular sampling.

    ``seed=0`` selects a seed from the current clock time; any other seed
    makes the generated ray distribution reproducible.
    """
    if beam_points < 1 or mirror_segments < 1 or rays_per_beam_point < 1:
        raise ValueError("beam_points, mirror_segments, and rays_per_beam_point must be positive")
    if intensity_sharpness < 0 or angular_spread_sharpness < 0:
        raise ValueError("Graph sharpness values must be non-negative")
    if any(fraction <= 0 or fraction > 1 for fraction in collection_fractions):
        raise ValueError("collection_fractions must be greater than 0 and at most 1")
    # A LaTeX expression is intentionally accepted as the public interchange
    # format.  Coefficients are recovered from the expression's polynomial
    # syntax without requiring sympy.
    expression = mirror_polynomial_latex.strip().replace("$", "").replace("y=", "")
    expression = expression.replace(" ", "").replace("{", "").replace("}", "")
    expression = re.sub(
        r"(\d*\.?\d+)\*10\^(-?\d+)",
        lambda match: f"{match.group(1)}e{match.group(2)}",
        expression,
    )
    # Parse terms explicitly rather than evaluating arbitrary text supplied as
    # the LaTeX expression.
    terms = re.findall(
        r"([+-]?)(\d*\.?\d+(?:e[+-]?\d+)?)(x(?:\^(\d+))?)?",
        expression,
        re.I,
    )
    if not terms:
        raise ValueError("mirror_polynomial_latex is not a supported polynomial expression")
    degree = max((int(power) if power else (1 if variable else 0)) for sign, number, variable, power in terms)
    coefficients = np.zeros(degree + 1)
    for sign, number, variable, power in terms:
        exponent = int(power) if power else (1 if variable else 0)
        coefficients[degree - exponent] += (-1 if sign == "-" else 1) * float(number)

    x_beam, y_beam, tangent_angles = _beam_points(bending_radius, bending_angle, beam_points)
    amplitude, peak_microrad, sigma_microrad = _fit_symmetric_gaussians(
        angular_distribution_points, angular_distribution_initial_guess
    )
    actual_seed = int(time.time_ns()) if seed == 0 else seed
    rng = np.random.default_rng(actual_seed)
    peak_selector = rng.integers(0, 2, size=(beam_points, rays_per_beam_point))
    peak_offsets = np.where(peak_selector == 0, -peak_microrad, peak_microrad)
    offsets_microrad = peak_offsets + rng.normal(
        0.0,
        sigma_microrad,
        size=(beam_points, rays_per_beam_point),
    )
    rays = np.repeat(
        np.column_stack((x_beam, y_beam, tangent_angles)),
        rays_per_beam_point,
        axis=0,
    )
    rays[:, 2] += offsets_microrad.ravel() * 1e-6
    domain = _projected_ray_domain(rays, coefficients, tuple(map(float, mirror_domain)))
    started = time.perf_counter()
    hit, hit_x, hit_y, reflected = _trace_polynomial(
        rays,
        coefficients,
        domain,
        show_progress=show_progress,
        progress_total=len(rays),
        progress_started=started,
    )
    trace_seconds = time.perf_counter() - started
    if show_progress:
        print(f"\r{' ' * 90}\r", end="")
        print(f"Tracing complete: {len(rays):,} rays in {trace_seconds:.2f}s "
              f"({len(rays) / max(trace_seconds, 1e-9):,.0f} rays/s)")

    result: dict[str, object] = {
        "mirror_polynomial_latex": mirror_polynomial_latex,
        "requested_domain": tuple(map(float, mirror_domain)),
        "tested_domain": domain,
        "rays": rays,
        "hit": hit,
        "hit_x": hit_x,
        "hit_y": hit_y,
        "reflected_directions": reflected,
        "hit_count": int(hit.sum()),
        "ray_count": int(len(rays)),
        "seed": actual_seed,
        "fitted_distribution_amplitude": amplitude,
        "fitted_distribution_peak_microrad": peak_microrad,
        "fitted_distribution_sigma_microrad": sigma_microrad,
        "trace_seconds": trace_seconds,
    }
    valid = hit.sum() > 0
    if valid:
        collection_distances = _central_collection_distances(
            hit, hit_x, hit_y, reflected, (focus_x, focus_y), collection_fractions
        )
        grid, density, peak_intensity = _screen_profile(
            hit,
            hit_x,
            hit_y,
            reflected,
            (focus_x, focus_y),
            sharpness=max(intensity_sharpness, 1e-9),
        )
        result.update(intensity_position_micrometers=grid, intensity_density=density,
                      peak_intensity=float(peak_intensity),
                      collection_distances_meters=collection_distances,
                      collection_distances_micrometers={
                          fraction: distance * 1e6
                          for fraction, distance in collection_distances.items()
                      })

    show_intensity = intensity_sharpness > 0 and valid
    show_angular_spread = angular_spread_sharpness > 0
    if show_model or show_intensity or show_angular_spread:
        import matplotlib.pyplot as plt
    if show_model:
        figure, axis = plt.subplots(figsize=(11, 7))
        arc = np.linspace(0, bending_angle, 400)
        axis.plot(bending_radius * np.sin(arc), bending_radius * (np.cos(arc) - 1), "k--", label="Beam arc")
        mirror_x = np.linspace(*domain, 2000)
        mirror_y = np.polynomial.polynomial.polyval(mirror_x, coefficients[::-1])
        axis.plot(mirror_x, mirror_y, "r", label="Mirror")
        draw = np.linspace(0, len(rays) - 1, min(len(rays), 3000), dtype=int)
        # A fixed alpha makes dense ray bundles look like opaque triangles.
        # Scale both values with the number of drawn rays so density remains
        # visible at low and high simulation resolutions.
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
        reflected_start = np.column_stack((hit_x[drawn], hit_y[drawn]))
        reflected_end = reflected_start + 30.0 * reflected[drawn]
        reflected_segments = np.stack((reflected_start, reflected_end), axis=1)
        axis.add_collection(LineCollection(
            incident_segments, colors="tab:blue", alpha=ray_alpha, linewidths=ray_width
        ))
        axis.add_collection(LineCollection(
            reflected_segments, colors="tab:orange", alpha=ray_alpha, linewidths=ray_width
        ))
        axis.plot(focus_x, focus_y, "ro", markersize=5, label="Focus")
        mean_direction = np.nanmean(reflected[hit], axis=0)
        mean_direction /= np.linalg.norm(mean_direction)
        screen_normal = np.array((-mean_direction[1], mean_direction[0]))
        screen_half_length = max(1.0, 0.08 * (domain[1] - domain[0]))
        screen_start = np.array((focus_x, focus_y)) - screen_half_length * screen_normal
        screen_end = np.array((focus_x, focus_y)) + screen_half_length * screen_normal
        axis.plot([screen_start[0], screen_end[0]], [screen_start[1], screen_end[1]],
                  color="black", linewidth=1.2, label="Screen")
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel("y (m)", fontsize=AXIS_FONT_SIZE)
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        axis.grid(True, linestyle=":")
        axis.legend()
    if show_intensity and valid:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(grid, density, color="purple")
        axis.set_xlabel("Screen displacement (micrometres)")
        axis.set_ylabel("Relative intensity density")
        axis.set_xlabel("Screen displacement (micrometres)", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel("Relative intensity density", fontsize=AXIS_FONT_SIZE)
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        axis.grid(True, linestyle=":")
    if show_angular_spread:
        figure, axis = plt.subplots(figsize=(10, 5))
        spread_grid, spread_density = _angular_profile(
            offsets_microrad.ravel(),
            sigma_microrad,
            angular_spread_sharpness,
        )
        axis.plot(
            spread_grid,
            spread_density,
            color="tab:blue",
            linewidth=2.5,
        )
        axis.fill_between(spread_grid, spread_density, color="tab:blue", alpha=0.2)
        axis.set_xlabel("Angular offset (microradians)")
        axis.set_ylabel("Frequency density")
        axis.set_xlabel("Angular offset (microradians)", fontsize=AXIS_FONT_SIZE)
        axis.set_ylabel("Frequency density", fontsize=AXIS_FONT_SIZE)
        axis.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        axis.grid(True, linestyle=":")
    if show_model or show_intensity or show_angular_spread:
        plt.show(block=False)
        plt.pause(0.001)
    return result
