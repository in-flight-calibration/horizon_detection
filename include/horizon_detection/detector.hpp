#pragma once

#include <ifc_tools/debug_viewer.hpp>
#include <horizon_detection/horizon.hpp>

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

    virtual Horizon detect(const rclcpp::Time& timestamp, const cv::Mat& image) = 0;

protected:
    Detector(const std::string& node_name) : rclcpp::Node(node_name) {}

    rclcpp::Time _last_update = rclcpp::Time(0);
    Horizon _last_horizon = {};

    std::optional<ifc_tools::DebugViewer> _viewer = std::nullopt;
};

}
