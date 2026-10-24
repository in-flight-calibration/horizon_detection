#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <utility>

namespace horizon {
    
struct Horizon
{
    float theta         = 0.0f;
    float rho           = 0.0f;
    float length        = 0.0f;
    float confidence    = 0.0f;
    
    float sigma_theta   = std::numeric_limits<float>::quiet_NaN();
    float sigma_rho     = std::numeric_limits<float>::quiet_NaN();
};

}
