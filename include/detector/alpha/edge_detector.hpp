#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace horizon::hough
{

struct EdgeDetectorParams
{
    double canny_high_scale = 2.0;
    double canny_low_scale = 0.8;
};

class EdgeDetector
{
public:
    explicit EdgeDetector(const EdgeDetectorParams& params)
        : _params(params)
    {
        if (!std::isfinite(_params.canny_high_scale) || _params.canny_high_scale <= 0.0
            || !std::isfinite(_params.canny_low_scale) || _params.canny_low_scale <= 0.0
            || _params.canny_high_scale < _params.canny_low_scale)
        {
            throw std::invalid_argument("Invalid edge detector parameter");
        }
    }

    bool apply(const cv::Mat& image, cv::Mat& out) const
    {
        if (image.type() != CV_8UC1)
        {
            return false;
        }

        cv::Mat gradient_x, gradient_y, magnitude, unused;
        cv::spatialGradient(image, gradient_x, gradient_y);
        cv::convertScaleAbs(gradient_x, gradient_x);
        cv::convertScaleAbs(gradient_y, gradient_y);
        cv::add(gradient_x, gradient_y, magnitude);

        const double otsu = cv::threshold(magnitude, unused, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
        const double high = std::clamp(_params.canny_high_scale * otsu, 0.0,  255.0);
        const double low = std::clamp(_params.canny_low_scale * otsu, 0.0, 255.0);
        cv::Canny(image, out, low, high);
        return true;
    }

private:
    EdgeDetectorParams _params;
};

}
