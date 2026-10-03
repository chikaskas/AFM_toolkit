# Arc-Focusing Mirror (AFM) User Guide

## Overview

This folder contains a simulation of a free-form Arc-Focusing Mirror (AFM) for synchrotron radiation from an electron beam travelling through a bending magnet. The program:

1. Constructs an ideal mirror polyline by bisecting the incoming beam
   direction and the direction from each mirror point to the focus.
2. Fits the resulting mirror points with a polynomial.
3. Samples divergent rays from every testing beam point.
4. Intersects the rays with the fitted polynomial, reflects them, and measures
   their distribution on a screen through the configured focus.

`ellip_runner.py` performs the same ray sampling, screen analysis, and graph
generation using an analytical elliptical mirror. This provides a comparison
whose mirror shape and reflection geometry are the only intended differences.

## Files

| File | Purpose |
|---|---|
| `AFM_module.py` | Importable implementation of `define_mirror()` and `test_mirror()`. |
| `AFM_runner.py` | Command-line program that defines and tests the AFM. |
| `ellip_runner.py` | Command-line comparison program for an elliptical mirror. |
| `config.json` | Shared configuration used by both runner programs. |

## Requirements

Use Python 3.10 or newer and install:

```bash
python -m pip install numpy scipy matplotlib
```

Run commands from this folder, or provide the full path to the folder.

On macOS or Linux, the complete setup can be run as:

```bash
python3 -m pip install numpy scipy matplotlib
python3 AFM_runner.py
```

If `python` and `pip` refer to different installations, use the same
interpreter for both commands, for example `python3 -m pip` and
`python3 AFM_runner.py`.

## Running the programs

The runners use `config.json` automatically when no argument is supplied:

```bash
python AFM_runner.py
python ellip_runner.py
```

An alternative JSON configuration can be supplied explicitly:

```bash
python AFM_runner.py path/to/other_config.json
python ellip_runner.py path/to/other_config.json
```

Both programs print the hit count, collection widths, and peak screen
intensity. If a graph is enabled, the terminal remains available until the
Matplotlib windows are closed.

For a quick performance test, temporarily use smaller values for
`defining_beam_points`, `testing_beam_points`, and `rays_per_beam_point`.
Restore the production values before comparing mirrors. The total number of
tested rays is:

```text
testing_beam_points * rays_per_beam_point
```

The AFM definition step uses `defining_beam_points` and is separate from the
ray-testing step. Increasing `defining_beam_points` improves sampling of the
ideal polyline but can substantially increase startup time. Increasing
`testing_beam_points` or `rays_per_beam_point` improves statistical stability
but increases ray-tracing time and memory use.

## Configuration reference

All distances are in metres unless stated otherwise. The beam starts at
`(0, 0)`, initially travels in the positive x direction, and bends toward
negative y. The first mirror point is `(mirror_distance, 0)`.

### Beam and focal geometry

| Key | Meaning |
|---|---|
| `bending_radius` | Radius of the electron beam path. |
| `bending_angle` | Total beam bend, in radians. |
| `mirror_distance` | x-coordinate of the first mirror point. |
| `focal_angle_degrees` | Focal direction angle in degrees, measured at the first mirror point from the direction back toward the beam origin. |
| `focal_distance` | Distance from the first mirror point to the configured focus. |
| `polynomial_degree` | Degree of the polynomial fitted to the constructed AFM points. |

The focus used by both programs is calculated as:

```text
focus_x = mirror_distance - focal_distance * cos(focal_angle_degrees)
focus_y = focal_distance * sin(focal_angle_degrees)
```

The elliptical program uses the origin `(0, 0)` as its first focus and this
calculated point as its second focus.

### Sampling and resolution

| Key | Meaning |
|---|---|
| `defining_beam_points` | Number of beam points used to construct the AFM polyline before polynomial fitting. Large values improve the fitted shape but increase definition time. |
| `testing_beam_points` | Number of beam points used for ray emission during testing. |
| `rays_per_beam_point` | Number of divergent rays emitted from each testing beam point. Total rays equal `testing_beam_points * rays_per_beam_point`. |
| `mirror_segments` | Retained compatibility setting. Current AFM tracing intersects rays directly with the fitted polynomial, so this value does not control the number of intersection segments. |

### Angular distribution

`angular_distribution_points` contains `[offset, density]` pairs. Offsets and
the fitted Gaussian parameters are in microradians. The program fits the
symmetric model:

```text
A * exp(-(x - B)^2 / (2 C^2)) + A * exp(-(x + B)^2 / (2 C^2))
```

where `A` is the amplitude, `B` is the positive and negative peak offset, and
`C` is the Gaussian standard deviation.

`angular_distribution_initial_guess` supplies the initial `[A, B, C]` values
for SciPy's nonlinear fit. It affects convergence, not the final model when
the fit converges to the same solution. If the supplied points contain only
one side and zero, the nonzero points are mirrored automatically.

Use offsets in microradians, not radians. For example, `10000` means
`10000 µrad = 0.010 rad`. The fitted angular offset is added to the local
beam tangent for every emitted ray.

### Output and graphs

| Key | Meaning |
|---|---|
| `seed` | Random seed for reproducible ray sampling. `0` uses a time-based seed. |
| `collection_fractions` | Fractions between 0 and 1 for which centered screen widths are reported. For example, `0.5` reports the central 50% width. |
| `show_model` | Shows the to-scale beam, mirror, incident rays, reflected rays, focus/foci, and screen. Axes are in metres. |
| `intensity_sharpness` | Enables the screen-intensity graph when greater than zero. Larger values increase the histogram resolution and reduce smoothing. `0` disables it. |
| `angular_spread_sharpness` | Enables the input angular-spread graph when greater than zero. Larger values increase resolution and reduce smoothing, subject to the common smoothing floor. `0` disables it. |

The intensity and angular graphs use histogram-based density estimates with
Gaussian smoothing. They are diagnostic plots; the reported collection widths
are calculated directly from the screen-ray coordinates.

Sharpness is not a physical parameter. It changes graph resolution and
smoothing only; it does not change the rays, mirror, hit count, or collection
widths. To compare two mirrors fairly, use the same seed and all the same
sampling and graph settings.

## Importing the AFM module

The public module API is:

```python
from AFM_module import define_mirror, test_mirror
```

`define_mirror()` accepts the beam geometry, focal coordinates, and polynomial
degree. It returns:

```python
(mirror_polynomial_latex, (domain_x_min, domain_x_max))
```

`test_mirror()` accepts the testing geometry, the returned polynomial and
domain, angular-distribution data, sampling settings, and graph settings. It
returns a dictionary containing the generated rays, hit locations, reflected
directions, hit counts, fitted angular parameters, and screen metrics.

The public signatures are:

```python
define_mirror(
    beam_points, bending_radius, bending_angle, mirror_distance,
    focus_x, focus_y, polynomial_degree, show_progress=False
)

test_mirror(
    beam_points, bending_radius, bending_angle, focus_x, focus_y,
    mirror_polynomial_latex, mirror_domain,
    rays_per_beam_point=25,
    angular_distribution_points=(),
    angular_distribution_initial_guess=None,
    seed=20260906,
    collection_fractions=(),
    mirror_segments=1000,
    show_model=False,
    intensity_sharpness=0.0,
    angular_spread_sharpness=0.0,
    max_domain_expansions=8,
    show_progress=False
)
```

The returned result dictionary includes `hit_count`, `ray_count`, `rays`,
`hit`, `hit_x`, `hit_y`, `reflected_directions`, the fitted distribution
parameters, and—when at least one ray hits—the screen intensity arrays and
collection widths in metres and micrometres.

For library use, keep plotting disabled and inspect the returned arrays:

```python
from AFM_module import define_mirror, test_mirror

polynomial, domain = define_mirror(
    beam_points=1000,
    bending_radius=11.5,
    bending_angle=0.032,
    mirror_distance=15.0,
    focus_x=15.0,
    focus_y=15.0,
    polynomial_degree=3,
)

result = test_mirror(
    beam_points=500,
    bending_radius=11.5,
    bending_angle=0.032,
    focus_x=15.0,
    focus_y=15.0,
    mirror_polynomial_latex=polynomial,
    mirror_domain=domain,
    angular_distribution_points=[[-10000, 1000], [-2500, 29150], [0, 24275]],
    seed=20260906,
    show_model=False,
)
print(result["hit_count"], result["ray_count"])
```

Angles passed to these Python functions are radians, except for angular
distribution offsets, which are microradians. The command-line runners handle
the degree-to-radian conversion for `focal_angle_degrees`.

## Comparing the mirrors

Use the same `config.json` with both commands:

```bash
python AFM_runner.py config.json
python ellip_runner.py config.json
```

Compare the collection widths, peak intensity, hit count, and graphs. The
elliptical mirror is not polynomial-fitted; its analytical ellipse and normal
are the only deliberate differences in the ray-intersection and reflection
steps.

Interpret results using the same sample size. A smaller reported collection
width indicates a tighter central focal spot. Peak intensity is a smoothed
density estimate and should not be compared across different
`intensity_sharpness`, ray counts, or screen sampling settings. A lower hit
count means that some generated rays did not intersect the tested mirror
surface and should be investigated before drawing performance conclusions.

## Troubleshooting

| Symptom | Likely cause or fix |
|---|---|
| `ModuleNotFoundError: AFM_module` | Run the command from this folder, or add this folder to the Python path. |
| `FileNotFoundError` for the configuration | Pass the configuration path explicitly, or ensure `config.json` is beside the runner. |
| Very slow startup | Reduce `defining_beam_points`; this affects AFM construction, not the ellipse. |
| No intensity graph | Set `intensity_sharpness` to a value greater than zero and ensure at least one ray hits. |
| No angular graph | Set `angular_spread_sharpness` to a value greater than zero. |
| Different results between runs | Use the same nonzero `seed`; `seed: 0` deliberately produces a time-based sequence. |
| Fit failure or implausible angular distribution | Check that the distribution points are `[offset, density]` pairs in microradians and provide a suitable initial guess. |

The graph windows are intentionally blocking at the end of each runner:
close them to return control to the terminal.
