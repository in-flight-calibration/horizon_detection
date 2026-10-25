#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace horizon::hough
{

struct MorphologicalClosingParams
{
    int closing_kernel_size = 7;
};

class MorphologicalClosing
{
public:
    explicit MorphologicalClosing(const MorphologicalClosingParams& params)
        : _params(params)
    {
        if (_params.closing_kernel_size <= 0 
            || _params.closing_kernel_size > 101
            || _params.closing_kernel_size % 2 == 0)
        {
            throw std::invalid_argument("Invalid morphological closing parameter");
        }

        _kernel = cv::getStructuringElement(
            cv::MORPH_RECT, 
            cv::Size(_params.closing_kernel_size, _params.closing_kernel_size)
        );
    }

    bool apply(const cv::Mat& image, cv::Mat& out) const
    {
        cv::morphologyEx(image, out, cv::MORPH_CLOSE, _kernel);
        return true;
    }

private:
    MorphologicalClosingParams _params;
    
    cv::Mat _kernel;
};

}
