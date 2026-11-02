#pragma once

#include <ifc_tools/debug_viewer.hpp>
#include <ifc_tools/performance_monitor.hpp>
#include <horizon.hpp>

#include <chrono>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <utility>

namespace horizon
{

class Detector : public rclcpp::Node
{
public:
    virtual ~Detector() = default;

    Horizon detect(const rclcpp::Time& image_timestamp, const cv::Mat& image)
    {
        _performance_monitor.start(get_clock()->now());
        const Horizon result = _detect(image_timestamp, image);
        _performance_monitor.stop(get_clock()->now());

        auto now = get_clock()->now();
        if ((now - _last_performance_monitor_update).nanoseconds() >= PERFORMANCE_MONITOR_INTERVAL_NS)
        {
            auto stats = _performance_monitor.statistics();
            RCLCPP_INFO(get_logger(), "Performance: count=%zu, avg=%ld, min=%ld, max=%ld", 
                stats.count,
                stats.avg_ns,
                stats.min_ns,
                stats.max_ns
            );

            _performance_monitor.reset();
            _last_performance_monitor_update = now;
        }

        return result;
    }

    static constexpr Horizon INVALID_HORIZON = {
        .theta = std::numeric_limits<float>::quiet_NaN(),
        .rho = std::numeric_limits<float>::quiet_NaN(),
        .confidence = 0.0f,
        .sigma_theta = std::numeric_limits<float>::quiet_NaN(),
        .sigma_rho = std::numeric_limits<float>::quiet_NaN()
    };

protected:
    static constexpr int64_t PERFORMANCE_MONITOR_INTERVAL_NS = 5'000'000'000; // 5s

    Detector(const std::string& node_name, const rclcpp::NodeOptions& options)
        : rclcpp::Node(node_name, options), _last_performance_monitor_update(get_clock()->now())
    {
    }

    virtual Horizon _detect(const rclcpp::Time& image_timestamp, const cv::Mat& image) = 0;

    ifc_tools::PerformanceMonitor _performance_monitor;
    rclcpp::Time _last_performance_monitor_update;

    std::optional<ifc_tools::DebugViewer> _viewer = std::nullopt;
};

}
