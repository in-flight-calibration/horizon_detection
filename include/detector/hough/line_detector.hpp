#pragma once

#include <horizon.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace horizon::hough
{

struct LineDetectorParams
{
    int threshold = 50;

    double rho_resolution_px = 3.0;
    double theta_resolution_degrees = 1.0;
    double minimum_line_length_ratio = 0.2;
    double maximum_line_gap_ratio = 0.025;
};

class LineDetector
{
public:
    explicit LineDetector(const LineDetectorParams& params)
        : _params(params)
    {
        if (!std::isfinite(_params.rho_resolution_px) ||
            !std::isfinite(_params.theta_resolution_degrees) ||
            _params.rho_resolution_px <= 0.0 ||
            _params.theta_resolution_degrees <= 0.0 ||
            !std::isfinite(_params.minimum_line_length_ratio) ||
            !std::isfinite(_params.maximum_line_gap_ratio) ||
            _params.threshold <= 0 ||
            _params.minimum_line_length_ratio <= 0.0 ||
            _params.minimum_line_length_ratio > 1.0 ||
            _params.maximum_line_gap_ratio < 0.0 ||
            _params.maximum_line_gap_ratio > 1.0)
        {
            throw std::invalid_argument("Invalid line detector parameter");
        }
    }

    bool detect(const cv::Mat& edges, std::vector<cv::Vec4i>& lines) const
    {
        if (edges.type() != CV_8UC1)
        {
            return false;
        }

        const double width = static_cast<double>(edges.cols);
        cv::HoughLinesP(
            edges,
            lines,
            _params.rho_resolution_px,
            _params.theta_resolution_degrees * std::numbers::pi / 180.0,
            _params.threshold,
            _params.minimum_line_length_ratio * width,
            _params.maximum_line_gap_ratio * width);

        return true;
    }

    static cv::Mat get_debug_view(cv::Size size, const std::vector<cv::Vec4i>& lines)
    {
        cv::Mat output = cv::Mat::zeros(size, CV_8UC1);

        for (const cv::Vec4i& line : lines)
        {
            cv::line(output, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(220), 1, cv::LINE_AA);
        }

        return output;
    }

private:
    LineDetectorParams _params;
};

}
