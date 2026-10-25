#pragma once

#include <detector/detector.hpp>
#include <detector/hough/detector_alpha.hpp>

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
        if (name == "alpha")
        {
            return std::make_unique<DetectorAlpha>();
        }

        throw std::invalid_argument("Unknown horizon detector: " + name);
    }
};

}
