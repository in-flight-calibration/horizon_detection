#pragma once

#include <detector/detector.hpp>
#include <detector/alpha/edge_detector.hpp>
#include <detector/alpha/horizon_estimator.hpp>
#include <detector/alpha/line_grouping.hpp>
#include <detector/alpha/morphological_closing.hpp>
#include <detector/alpha/line_detector.hpp>
#include <detector/alpha/prefilter.hpp>

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

protected:
    Horizon _detect(const rclcpp::Time& image_timestamp, const cv::Mat& image) override
    {
        if (image.empty())
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Received an empty image");
            return {};
        }

        _display_utils.update(image.size());

        _state.image = image;

        if (!_prefilter.apply(_state.image, _state.filtered))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Hough processing failed");
            return {};
        }

        _utils.update(_state.filtered.size());

        if (!_edge_detector.apply(_state.filtered, _state.edges)
            || !_closing.apply(_state.edges, _state.closed)
            || !_line_detector.detect(_state.closed, _state.lines)
            || !_line_grouping.group(_state.lines, _utils, _state.groups))
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Hough processing failed");
            return {};
        }

        _state.estimate = _estimator.estimate(image_timestamp.nanoseconds(), _state.groups);

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
        std::vector<hough::LineGroup> groups;
        Horizon estimate;
    } _state;

    hough::Prefilter _prefilter;
    hough::EdgeDetector _edge_detector;
    hough::MorphologicalClosing _closing;
    hough::LineDetector _line_detector;
    hough::LineGrouping _line_grouping;
    hough::HorizonEstimator _estimator;

    HorizonUtils _utils;
    HorizonUtils _display_utils;

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
        params.downscale = declare_parameter<int>("horizon.alpha.prefilter.downscale", params.downscale);
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
        params.distance_tolerance = declareFloat(
            "horizon.alpha.line_grouping.distance_tolerance", params.distance_tolerance);
        params.minimal_coverage = declareFloat(
            "horizon.alpha.line_grouping.minimal_coverage", params.minimal_coverage);
        params.slots = declare_parameter<int>("horizon.alpha.line_grouping.slots", params.slots);
        return params;
    }

    hough::HorizonEstimatorParams declareHorizonEstimatorParams()
    {
        hough::HorizonEstimatorParams params;
        params.minimal_init_score = declareFloat(
            "horizon.alpha.estimator.minimal_init_score", params.minimal_init_score);
        params.minimal_update_score = declareFloat(
            "horizon.alpha.estimator.minimal_update_score", params.minimal_update_score);
        params.timeout_s = declareFloat(
            "horizon.alpha.estimator.timeout_s", params.timeout_s);
        params.max_theta_rate = declareFloat(
            "horizon.alpha.estimator.max_theta_rate", params.max_theta_rate);
        params.max_rho_rate = declareFloat(
            "horizon.alpha.estimator.max_rho_rate", params.max_rho_rate);
        params.confidance_filter_tau = declareFloat(
            "horizon.alpha.estimator.confidance_filter_tau", params.confidance_filter_tau);
        return params;
    }

    void showDebugView()
    {
        auto updateTile = [this](size_t index, cv::Mat tile, const char* label)
        {
            _viewer->update(index / _debug_cols, index % _debug_cols, tile, label);
        };

        const cv::Mat line_detector_view = hough::LineDetector::getDebugView(_state.filtered.size(), _state.lines);
        const cv::Mat line_grouping_view = hough::LineGrouping::getDebugView(
            _state.filtered.size(), _state.groups, _utils);
        const cv::Mat estimated_horizon_view = hough::HorizonEstimator::getDebugView(
            _state.image, _state.estimate, _display_utils);

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
