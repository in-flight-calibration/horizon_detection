#pragma once

#include <ifc_tools/debug_viewer.hpp>
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
        return _detect(image_timestamp, image);
    }

    static constexpr Horizon INVALID_HORIZON = {
        .theta = std::numeric_limits<float>::quiet_NaN(),
        .rho = std::numeric_limits<float>::quiet_NaN(),
        .confidence = 0.0f,
        .sigma_theta = std::numeric_limits<float>::quiet_NaN(),
        .sigma_rho = std::numeric_limits<float>::quiet_NaN()
    };

protected:
    Detector(const std::string& node_name, const rclcpp::NodeOptions& options)
        : rclcpp::Node(node_name, options)
    {
    }

    virtual Horizon _detect(const rclcpp::Time& image_timestamp, const cv::Mat& image) = 0;

    std::optional<ifc_tools::DebugViewer> _viewer = std::nullopt;
};

}
