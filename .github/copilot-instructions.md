# Copilot Instructions

## Overview

`horizon_detection` is a ROS 2 `ament_cmake` C++20 package in the colcon workspace at `/home/wgajda/dev/ifc`. The package directory is its own git repository. Its detector classes and node are header-only; only `src/main.cpp` is compiled.

- `horizon_detection_node` runs `horizon::HorizonDetection`, which owns an `ifc_tools::ImageReceiver`, a `Detector` pointer, and the ROS publisher. `main` currently adds only the receiver to its executor.
- `Detector` inherits from `rclcpp::Node`, stores temporal state and an optional `DebugViewer`, and exposes `detect(timestamp, image)`.
- `DetectorFactory::create(name)` constructs the configured implementation. `DetectorHough::detect()` currently returns a default result; the Hough algorithm is deferred.
- Read `PLAN.md` and the workspace-root `PLAN.md` for the horizon convention and staged Hough implementation. The reference paper is the PDF in the package root.
- `Horizon` is a simple struct of float fields: `theta`, `rho`, `length`, `confidence`, `sigma_theta`, and `sigma_rho`. It has no `valid` field; zero confidence denotes an invalid result.
- `Horizon.msg` contains `Header`, `theta`, `rho`, `confidence`, `theta_sd`, and `sigma_rho`. `HorizonDetection::publish()` maps those fields and propagates the image timestamp; internal `length` is not published.
- Build flags: `-Wall -Wextra -Wpedantic` on GCC/Clang — keep code warning-free.

## Build / run (from the workspace root `/home/wgajda/dev/ifc`)

```bash
source /opt/ros/lyrical/setup.bash
colcon build --packages-up-to horizon_detection
source install/setup.bash
ros2 run horizon_detection horizon_detection_node --ros-args --params-file config.yaml
./ifc.sh
```

`./ifc.sh` expects to run from the workspace root and builds the whole workspace before running the node. Image delivery requires FlightGear and the configured ROS middleware router. Debug windows require a desktop and are disabled by default.

## Conventions

- Source files use 4-space indentation, with class members indented one extra level under `public:`.
- New executables must be added to both `add_executable` and the `install(TARGETS ... DESTINATION lib/${PROJECT_NAME})` block so `ros2 run` finds them.
- ROS parameters and message conversion are in `HorizonDetection`. Preserve the source image timestamp when publishing.
