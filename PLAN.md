# PLAN: horizon_detection

## Problem and approach
`horizon_detection` (ROS 2 lyrical, C++20, OpenCV 4.10) has a header-only ROS node/detector scaffold.
`DetectorHough` currently returns a default result. The goal is to receive images from an aircraft's forward-facing camera, detect the horizon with pure-OpenCV implementations behind a common interface, and visualize intermediate stages.

Image source: `simulator/src/flightgear/src/video_node.cpp` publishes via `image_transport` on the base topic
`/flightgear/image` (`bgr8`, `SensorDataQoS`) → `/flightgear/image/compressed` = the `compressed` transport.

Scope:
- **New package `src/ifc_tools`** (shared workspace tools): `ImageReceiver` (ROS node) and `DebugViewer` (pure OpenCV) – see `../ifc_tools/PLAN.md`.
- **`horizon_detection`**: simple `Horizon` result struct, `Detector` base node, `DetectorFactory`, `DetectorHough`,
  and a `HorizonDetection` node that owns the image receiver, detector pointer, and publisher. Hough processing is deferred.
- `ifc_msgs/msg/Horizon.msg` carries the published estimate. ROS parameters configure the detector and nodes from the workspace `config.yaml`.
- `Detector` and `HorizonDetection` inherit from `rclcpp::Node`; `ImageReceiver` is also an `rclcpp::Node`. The horizon detection headers form an `INTERFACE` target; only `main.cpp` is compiled into the executable.

## Horizon definition (agreed)
- Normalized frame: origin at the image center, `u` to the right, `v` **up**, isotropic scale `D = ½·√(W²+H²)`:
  `u = (x − W/2)/D`, `v = (H/2 − y)/D` (corners lie on the unit circle; independent of resolution/aspect ratio).
- Line (Hesse normal form): `−u·sinθ + v·cosθ = ρ`.
  - `θ` [rad] ∈ (−π/2, π/2], CCW from horizontal; the line is undirected (no inverted flight assumed; contour-based methods cannot tell sky from ground).
  - `ρ > 0` ⇒ horizon above the center; every line crossing the frame has `|ρ| ≤ 1`; vertical lines and lines outside the frame are representable.
  - Normalization: θ out of range → `θ ± π`, `ρ → −ρ`.
- Attitude (documentation): roll ≈ θ, pitch ≈ −atan(ρ·D / f_px). Matches Eq. (9) of the paper
  (φ = atan(k·fx/fy), θ = atan((v0 − k·u0 − b)·cosφ / fy) for y = kx + b in px, y axis down); the Hesse form has no
  singularity at φ = ±π/2, which the paper handles separately (Eq. 10).

## API (namespace `horizon`)
```cpp
struct Horizon {
  float theta = 0.0f;
  float rho = 0.0f;
  float length = 0.0f;
  float confidence = 0.0f;
  float sigma_theta = NaN;
  float sigma_rho = NaN;
};
class Detector : public rclcpp::Node {
public:
  virtual ~Detector() = default;
  virtual Horizon detect(const rclcpp::Time& timestamp, const cv::Mat& image) = 0;
protected:
  explicit Detector(const std::string& node_name);
  rclcpp::Time _last_update;
  Horizon _last_horizon;
  std::optional<ifc_tools::DebugViewer> _viewer;
};
```
`Detector` is currently an `rclcpp::Node`; its base constructor takes a node name. It holds temporal state and an optional viewer, but the temporal state is not yet read or updated, and viewer construction/use is not wired to a ROS parameter.

The current `Horizon` is a simple float-field struct with no `valid` member. A zero confidence denotes an invalid result. The ROS message contains:

```text
std_msgs/Header header
float32 theta
float32 rho
float32 confidence
float32 theta_sd
float32 sigma_rho
```

`theta` is in radians in `[-π/2, π/2]`; `rho` is normalized by half the image diagonal and its documented message range is `[-1, 1]`. The message uses `u*sin(theta) - v*cos(theta) = -rho`, equivalent to the Hesse form above. `header.stamp` is the source image timestamp. `theta_sd` and `sigma_rho` carry the angular and distance standard deviations. The message does not include `valid` or `length`; `confidence == 0` denotes a failed detection.

## DetectorHough (deferred implementation)
`DetectorHough::detect(timestamp, image)` currently returns a default `Horizon`. Implement its image-processing stages later; lengths/distances should be relative to image size.
0. *(optional future stage)* image enhancement from the paper: Retinex (illumination compensation)
   + dark channel prior (defogging); Otsu sky/ground segmentation (edges only from the region boundary).
1. **Gray + denoising:** `cvtColor` → **bilateral** filter (default, edge-preserving – the paper's "improved Canny")
   or `GaussianBlur` (selectable in Params); optional downscale to a working width (the result is normalized anyway).
2. **Canny** (thresholds from Params).
3. **Morphological closing** `morphologyEx(MORPH_CLOSE)` to bridge small gaps.
4. **HoughLinesP** → segments (minLineLength, maxLineGap relative to width).
5. **Grouping:** each segment → (θ, ρ, length) in the normalized frame; greedy clustering starting from the longest
   (|Δθ| mod π < tolθ, |Δρ| < tolρ w.r.t. the length-weighted mean; handle θ wrap-around at ±π/2 with ρ sign flip).
6. **Group selection:** score = total length × span (extent of endpoint projections onto the line direction / length of the line's chord in the frame).
7. **Line fit:** points sampled along the group's segments → orthogonal regression (`cv::fitLine`, DIST_HUBER – works for vertical lines too)
   → normalized (θ, ρ); from residuals σρ ≈ σ/√N, σθ ≈ σ/(√N·spread along the line).
8. **Verification:**
   - edge continuity: fraction of samples along the fitted line (inside the frame) with an edge pixel within ≤ k px,
   - geometric support: chord span/coverage and total group length,
   - temporal consistency against the previous result stored in `Detector::_last_horizon` and `_last_update`,
   - `confidence` = combination (e.g. product) of normalized measures; `confidence == 0` denotes invalid.
- **Debug stages:** input, gray/blur, Canny, closed, Hough segments, groups (colored, best highlighted), result + support points.

## Structure / build
- `include/horizon_detection/{horizon,detector,detector_factory,detector_hough,horizon_detection}.hpp`;
  `src/main.cpp` creates `HorizonDetection` and adds its receiver node to the executor (executable: `horizon_detection_node`).
- CMake: export a header-only `INTERFACE` target; `main.cpp` links it and the `ifc_tools`, OpenCV, `rclcpp`, and `ifc_msgs` dependencies.
- package.xml: `ifc_tools`, `ifc_msgs`, `rclcpp`, `libopencv-dev` (the `px4_msgs` dependency has been removed; the `src/px4_msgs` package stays in the workspace).
- Build: `colcon build --packages-up-to horizon_detection` (also builds `ifc_tools`).

## Reference
Liu X. et al., "High-precision pure vision horizontal attitude measurement method for UAVs", *Measurement* 272 (2026) 121032
(`horizon_detection/1-s2.0-S0263224126007414-main.pdf`), Sec. 3.1 (horizon detection) and Sec. 4 (Δk, ΔL validation).

## Plan files in the repository
- `src/ifc_tools/PLAN.md` – `ifc_tools` package (ImageReceiver, DebugViewer).
- `src/horizon_detection/PLAN.md` – horizon definition, current API/scaffold, and deferred `DetectorHough` implementation.

## Current state and next steps
- ROS parameters are `image.topic`, `image.transport`, `horizon.detector`, and `horizon.topic`.
- `DetectorFactory::create(name)` supports `"hough"` and throws for unknown names.
- `HorizonDetection::onImage()` calls `detect(timestamp, image)` and then `publish(result, timestamp)`.
- `HorizonDetection::publish()` maps theta, rho, confidence, sigma_theta, and sigma_rho to the ROS message and preserves the source image timestamp. `length` is internal.
- `DetectorHough` currently returns `{}`. Implement the Hough stages above as the next algorithm work.
- The executor in `main.cpp` currently adds only the receiver node; `HorizonDetection` and detector nodes are not added.
- Tune thresholds on the real FlightGear stream after implementation (visually via DebugViewer).
- Keep `.github/copilot-instructions.md` and the workspace-root `AGENTS.md` synchronized if package architecture or build commands change.
