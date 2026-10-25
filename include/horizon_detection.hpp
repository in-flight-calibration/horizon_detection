#pragma once

#include <detector/detector_factory.hpp>
#include <detector/detector.hpp>
#include <horizon.hpp>
#include <ifc_tools/image_receiver.hpp>
#include <ifc_msgs/msg/horizon.hpp>
#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace horizon
{

class HorizonDetection final : public rclcpp::Node
{
public:
    explicit HorizonDetection(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node("horizon_detection", options)
    {
        const std::string image_topic = declare_parameter<std::string>("image.topic", "/flightgear/image");
        const std::string image_transport = declare_parameter<std::string>("image.transport", "compressed");
        const std::string detector_name = declare_parameter<std::string>("horizon.detector", "alpha");
        const std::string horizon_topic = declare_parameter<std::string>("horizon.topic", "/horizon_detection/horizon");

        _receiver = std::make_shared<ifc_tools::ImageReceiver>("horizon_detection_receiver", image_topic, image_transport);
        _detector = DetectorFactory::create(detector_name);
        _publisher = create_publisher<ifc_msgs::msg::Horizon>(horizon_topic, rclcpp::ServicesQoS());
        
        _receiver->setCallback(std::bind(&HorizonDetection::onImage, this, std::placeholders::_1, std::placeholders::_2));
    }

    const std::shared_ptr<ifc_tools::ImageReceiver>& receiver() const
    {
        return _receiver;
    }

private:
    std::shared_ptr<ifc_tools::ImageReceiver> _receiver;
    std::unique_ptr<Detector> _detector;
    rclcpp::Publisher<ifc_msgs::msg::Horizon>::SharedPtr _publisher;

    void onImage(const rclcpp::Time& timestamp, const cv::Mat& image) {
        if (!_detector) {
            return;
        }

        Horizon horizon = _detector->detect(timestamp, image);
        publish(horizon, timestamp);
    }

    void publish(const Horizon& horizon, const rclcpp::Time& timestamp) const
    {
        ifc_msgs::msg::Horizon message;
        message.header.stamp = static_cast<builtin_interfaces::msg::Time>(timestamp);
        message.theta = horizon.theta;
        message.rho = horizon.rho;
        message.confidence = horizon.confidence;
        message.theta_sd = horizon.sigma_theta;
        message.sigma_rho = horizon.sigma_rho;
        _publisher->publish(message);
    }
};

}
