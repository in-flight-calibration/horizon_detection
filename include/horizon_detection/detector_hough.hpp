#pragma once

#include <horizon_detection/detector.hpp>

namespace horizon
{

class DetectorHough final : public Detector
{
public:
    explicit DetectorHough()
        : Detector("horizon_detector_hough")
    {
    }

    Horizon detect(const rclcpp::Time& timestamp, const cv::Mat& image) override
    {
        static_cast<void>(timestamp);
        static_cast<void>(image);
        return {};
    }
};

}
