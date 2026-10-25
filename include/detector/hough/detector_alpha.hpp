#pragma once

#include <detector/detector.hpp>
#include <detector/hough/edge_detector.hpp>
#include <detector/hough/horizon_estimator.hpp>
#include <detector/hough/line_grouping.hpp>
#include <detector/hough/morphological_closing.hpp>
#include <detector/hough/line_detector.hpp>
#include <detector/hough/prefilter.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace horizon
{

class DetectorAlpha final : public Detector
{
public:
    explicit DetectorAlpha(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Detector("horizon_detector_alpha", options),
          _prefilter(declarePrefilterParams()),
          _edge_detector(declareEdgeDetectorParams()),
          _closing(declareMorphologicalClosingParams()),
          _line_detector(declareLineDetectorParams()),
          _line_grouping(declareLineGroupingParams()),
          _estimator(declareHorizonEstimatorParams())
    {
        if (declare_parameter<bool>("horizon.debug_view", false))
        {
            _viewer.emplace(cv::Size(1280, 720), _debug_rows, _debug_cols);
        }
    }

    Horizon detect(const rclcpp::Time& timestamp, const cv::Mat& image) override
    {
        if (image.empty())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Received an empty image");
            return {};
        }

        _utils.update(image.size());

        _state.image = image;

        if (!_prefilter.apply(_state.image, _state.filtered)
            || !_edge_detector.apply(_state.filtered, _state.edges)
            || !_closing.apply(_state.edges, _state.closed)
            || !_line_detector.detect(_state.closed, _state.lines)
            || !_line_grouping.group(_state.lines, _utils, _state.groups))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Hough processing failed");
            return {};
        }

        _state.estimate = _estimator.estimate(timestamp.nanoseconds(), _state.groups);

        if (_viewer)
        {
            showDebugView();
        }

        return _state.estimate;
    }

private:
    static constexpr size_t _debug_rows = 2;
    static constexpr size_t _debug_cols = 3;

    struct State
    {
        cv::Mat image;
        cv::Mat filtered;
        cv::Mat edges;
        cv::Mat closed;
        std::vector<cv::Vec4i> lines;
        hough::LineGroups groups;
        Horizon estimate;
    } _state;

    hough::Prefilter _prefilter;
    hough::EdgeDetector _edge_detector;
    hough::MorphologicalClosing _closing;
    hough::LineDetector _line_detector;
    hough::LineGrouping _line_grouping;
    hough::HorizonEstimator _estimator;

    HorizonUtils _utils;

    float declareFloat(const std::string& name, float default_value)
    {
        return static_cast<float>(declare_parameter<double>(name, static_cast<double>(default_value)));
    }

    hough::PrefilterParams declarePrefilterParams()
    {
        hough::PrefilterParams params;
        params.bilateral_diameter = declare_parameter<int>("horizon.alpha.prefilter.bilateral_diameter", params.bilateral_diameter);
        params.bilateral_sigma_color = declare_parameter<double>("horizon.alpha.prefilter.bilateral_sigma_color", params.bilateral_sigma_color);
        params.bilateral_sigma_space = declare_parameter<double>("horizon.alpha.prefilter.bilateral_sigma_space", params.bilateral_sigma_space);
        params.gaussian_kernel_size = declare_parameter<int>("horizon.alpha.prefilter.gaussian_kernel_size", params.gaussian_kernel_size);
        return params;
    }

    hough::EdgeDetectorParams declareEdgeDetectorParams()
    {
        hough::EdgeDetectorParams params;
        params.canny_high_scale = declare_parameter<double>("horizon.alpha.edge_detector.canny_high_scale", params.canny_high_scale);
        params.canny_low_scale = declare_parameter<double>("horizon.alpha.edge_detector.canny_low_scale", params.canny_low_scale);
        return params;
    }

    hough::MorphologicalClosingParams declareMorphologicalClosingParams()
    {
        hough::MorphologicalClosingParams params;
        params.closing_kernel_size = declare_parameter<int>("horizon.alpha.closing.kernel_size", params.closing_kernel_size);
        return params;
    }

    hough::LineDetectorParams declareLineDetectorParams()
    {
        hough::LineDetectorParams params;
        params.rho_resolution_px = declare_parameter<double>("horizon.alpha.line_detect.rho_resolution_px", params.rho_resolution_px);
        params.theta_resolution_degrees = declare_parameter<double>("horizon.alpha.line_detect.theta_resolution_degrees", params.theta_resolution_degrees);
        params.threshold = declare_parameter<int>("horizon.alpha.line_detect.threshold", params.threshold);
        params.minimum_line_length_ratio = declare_parameter<double>("horizon.alpha.line_detect.minimum_line_length_ratio", params.minimum_line_length_ratio);
        params.maximum_line_gap_ratio = declare_parameter<double>("horizon.alpha.line_detect.maximum_line_gap_ratio", params.maximum_line_gap_ratio);
        return params;
    }

    hough::LineGroupingParams declareLineGroupingParams()
    {
        hough::LineGroupingParams params;
        params.angle_tolerance_degrees = declareFloat(
            "horizon.alpha.line_grouping.angle_tolerance_degrees", params.angle_tolerance_degrees);
        params.rho_tolerance = declareFloat("horizon.alpha.line_grouping.rho_tolerance", params.rho_tolerance);
        params.slots = declare_parameter<int>("horizon.alpha.line_grouping.slots", params.slots);
        return params;
    }

    hough::HorizonEstimatorParams declareHorizonEstimatorParams()
    {
        hough::HorizonEstimatorParams params;
        params.minimum_coverage = declareFloat(
            "horizon.alpha.estimator.minimum_coverage", params.minimum_coverage);
        params.minimum_score = declareFloat(
            "horizon.alpha.estimator.minimum_score", params.minimum_score);
        params.ambiguity_margin = declareFloat(
            "horizon.alpha.estimator.ambiguity_margin", params.ambiguity_margin);
        params.stale_seconds = declareFloat(
            "horizon.alpha.estimator.stale_seconds", params.stale_seconds);
        params.max_theta_rate_degrees_per_second = declareFloat(
            "horizon.alpha.estimator.max_theta_rate_degrees_per_second",
            params.max_theta_rate_degrees_per_second);
        params.max_rho_rate_per_second = declareFloat(
            "horizon.alpha.estimator.max_rho_rate_per_second", params.max_rho_rate_per_second);
        params.tau_rise_seconds = declareFloat(
            "horizon.alpha.estimator.tau_rise_seconds", params.tau_rise_seconds);
        params.minimum_confidence = declareFloat(
            "horizon.alpha.estimator.minimum_confidence", params.minimum_confidence);
        params.switch_confirmation_frames = declare_parameter<int>(
            "horizon.alpha.estimator.switch_confirmation_frames", params.switch_confirmation_frames);
        return params;
    }

    void showDebugView()
    {
        auto updateTile = [this](size_t index, cv::Mat tile, const char* label)
        {
            _viewer->update(index / _debug_cols, index % _debug_cols, tile, label);
        };

        const cv::Mat line_detector_view = hough::LineDetector::get_debug_view(_state.image.size(), _state.lines);
        const cv::Mat line_grouping_view = hough::LineGrouping::get_debug_view(
            _state.image.size(), _state.groups, _utils);
        const cv::Mat estimated_horizon_view = hough::HorizonEstimator::get_debug_view(
            _state.image, _state.estimate, _utils);

        updateTile(0, _state.filtered, "1:FILTERED");
        updateTile(1, _state.edges, "2:EDGE DETECTED");
        updateTile(2, _state.closed, "3:MORPHO. CLOSED");
        updateTile(3, line_detector_view, "4:LINE DETECTED");
        updateTile(4, line_grouping_view, "5:LINE GROUPED");
        updateTile(5, estimated_horizon_view, "6:ESTIMATED HORIZON");

        const uint64_t timestamp_us = static_cast<uint64_t>(get_clock()->now().nanoseconds() / 1000);
        _viewer->render(timestamp_us);
    }
};

}
