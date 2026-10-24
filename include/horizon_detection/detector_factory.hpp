#pragma once

#include <horizon_detection/detector.hpp>
#include <horizon_detection/detector_hough.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace horizon
{

class DetectorFactory
{
public:
    static std::unique_ptr<Detector> create(const std::string& name)
    {
        if (name == "hough")
        {
            return std::make_unique<DetectorHough>();
        }

        throw std::invalid_argument("Unknown horizon detector: " + name);
    }
};

}
