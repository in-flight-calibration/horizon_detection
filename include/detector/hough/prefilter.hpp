#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace horizon::hough
{

struct PrefilterParams
{
    int bilateral_diameter = 11;
    double bilateral_sigma_color = 75.0;
    double bilateral_sigma_space = 75.0;
    int gaussian_kernel_size = 7;
};

class Prefilter
{
public:
    explicit Prefilter(const PrefilterParams& params)
        : _params(params)
    {
        if (_params.bilateral_diameter <= 0 
            || _params.bilateral_sigma_color <= 0.0 
            || _params.bilateral_sigma_space <= 0.0
            || _params.gaussian_kernel_size <= 0
            || _params.gaussian_kernel_size % 2 == 0)
        {
            throw std::invalid_argument("Invalid prefilter parameter");
        }
    }

    bool apply(const cv::Mat& image, cv::Mat& output) const
    {
        if (!check(image))
        {
            return false;
        }

        cv::Mat gray = toGray(image);

        cv::Mat smoothed;

        cv::bilateralFilter(
            gray,
            smoothed,
            _params.bilateral_diameter,
            _params.bilateral_sigma_color,
            _params.bilateral_sigma_space);

        cv::GaussianBlur(
            smoothed,
            output,
            cv::Size(_params.gaussian_kernel_size, _params.gaussian_kernel_size),
            0.0);

        return true;
    }

private:
    PrefilterParams _params;

        bool check(const cv::Mat& image) const
    {
        return image.depth() == CV_8U 
            && (image.channels() == 1 || image.channels() == 3 || image.channels() == 4);
    }

    cv::Mat toGray(const cv::Mat& image) const
    {
        cv::Mat gray;
        if (image.channels() == 1)
        {
            gray = image;
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        }
        else
        {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        return gray;
    }
};

}
