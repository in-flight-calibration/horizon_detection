# Copilot Instructions

## Overview

`horizon_detection` is a ROS 2 `ament_cmake` C++20 package in the colcon workspace at `/home/wgajda/dev/ifc`. The package directory is its own git repository. Detector classes and the node are header-only; only `src/main.cpp` is compiled.

- `horizon_detection_node` runs `horizon::HorizonDetection`, which owns an `ifc_tools::ImageReceiver`, a `Detector` pointer, and the ROS publisher. `main` adds only the receiver to its executor.
- `Detector` inherits `rclcpp::Node`, stores optional `DebugViewer`, and exposes `detect(timestamp, image)`.
- `DetectorFactory::create(name)` constructs the configured implementation. `DetectorAlpha` runs the stages below and is selected with `horizon.detector: alpha`.
- `horizon.debug_view` defaults to `false`; when enabled, the detector displays a labeled 2×3 mosaic: filtered image, Canny, closing, Hough segments, grouped candidates, final horizon.
- Stage headers are under `include/detector/hough/` in `horizon::hough`; shared line math is `HorizonUtils` in `horizon.hpp`. Geometry uses `float`.
- `HorizonEstimator::estimate(timestamp_ns, groups)` returns a `Horizon` directly. It scores groups by coverage, how much other line length is below them, and rate-limited closeness to its track. Implausible jumps are rejected; the last line is held through short (< `stale_seconds`) candidate gaps. Confidence accumulation and score ambiguity handling are internal. See the workspace-root `PLAN.md` for parameters and current behavior.
- `Horizon` fields are `theta`, `rho`, `length`, `confidence`, `sigma_theta`, `sigma_rho`; there is no `valid` field and zero confidence denotes no detection.
- `Horizon.msg` fields are `header`, `image_timestamp`, `theta`, `rho`, `confidence`, `theta_sd`, and `sigma_rho`. `header.stamp` is publication time; `image_timestamp` preserves the source image timestamp. Internal `length` is not published.
- `examples/synthetic_horizon.cpp` animates a horizon and a perspective runway with two edges converging at the horizon. A headless, bounded test is `ros2 run horizon_detection synthetic_horizon --ros-args -p debug_view:=false -p duration:=30.0`; it exits nonzero if detection/accuracy criteria fail. If ROS initialization hangs, unset `ZENOH_ROUTER_CHECK_ATTEMPTS` or set it to `1` (`0` waits forever for a router).
- Build flags: `-Wall -Wextra -Wpedantic` — keep code warning-free.

## Horizon convention

- Normalized frame: origin at image centre, `u` right, `v` up, `D = ½·√(W²+H²)`, `u = (x − W/2)/D`, `v = (H/2 − y)/D`.
- Hesse form: `−u·sinθ + v·cosθ = ρ`. Normal `n = (−sinθ, cosθ)` points image-up; direction `d = (cosθ, sinθ)`.
- `θ` ∈ (−π/2, π/2], CCW from horizontal. `normalizeLine` and `alignLine` apply θ ± π with `ρ → −ρ`. Positive `ρ` means above image centre. Inverted flight is not assumed.

## Hough stages

1. `Prefilter`: grayscale, bilateral filter, Gaussian blur.
2. `EdgeDetector`: Canny thresholds scaled from Otsu threshold of `spatialGradient` magnitude.
3. `MorphologicalClosing`: bridges small gaps using `MORPH_CLOSE`.
4. `LineDetector`: `HoughLinesP`; minimum length and maximum gap are relative to image width.
5. `LineGrouping`: greedily clusters normalized segments (longest first) into at most `slots` groups, fits a length-weighted TLS line per group, computes coverage as union of segment projections / visible length, and sorts groups by coverage.
6. `HorizonEstimator`: chooses a candidate and accumulates confidence.

All stage parameters are ROS parameters under `horizon.alpha.*`; header defaults match workspace-root `config.yaml`.

## Build / run (from the workspace root)

```bash
source /opt/ros/lyrical/setup.bash
colcon build --packages-up-to horizon_detection
source install/setup.bash
ros2 run horizon_detection horizon_detection_node --ros-args --params-file config.yaml
./ifc.sh
```

Image delivery requires FlightGear and the configured ROS middleware router. Debug windows require a desktop.

## Conventions

- Four-space indentation, Allman braces; private members start with `_`.
- Preserve source image timestamps in `image_timestamp` when publishing.
- New executables must be added to `add_executable` and `install(TARGETS ... DESTINATION lib/${PROJECT_NAME})`.
