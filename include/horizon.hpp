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

class HorizonUtils
{
public:
    HorizonUtils() = default;

    explicit HorizonUtils(cv::Size size)
    {
        update(size);
    }

    void update(cv::Size size)
    {
        if (size == _size)
        {
            return;
        }

        _size = size;
        _center = cv::Point2f(static_cast<float>(size.width) / 2.0f, static_cast<float>(size.height) / 2.0f);
        _scale = 0.5f * std::hypot(static_cast<float>(size.width), static_cast<float>(size.height));
        _half_extent = cv::Point2f(
            static_cast<float>(size.width) / (2.0f * _scale),
            static_cast<float>(size.height) / (2.0f * _scale));
    }

    cv::Size size() const
    {
        return _size;
    }

    float scale() const
    {
        return _scale;
    }

    cv::Point2f toNormalized(const cv::Point2f& pixel) const
    {
        return cv::Point2f((pixel.x - _center.x) / _scale, (_center.y - pixel.y) / _scale);
    }

    cv::Point toPixel(const cv::Point2f& point) const
    {
        return cv::Point(
            static_cast<int>(std::lround(point.x * _scale + _center.x)),
            static_cast<int>(std::lround(_center.y - point.y * _scale)));
    }

    bool endpoints(const Horizon& horizon, cv::Point& first, cv::Point& second) const
    {
        const cv::Point2f foot = lineNormal(horizon.theta) * horizon.rho;
        const cv::Point2f half_chord = lineDirection(horizon.theta) * (chordLength(horizon.rho) / 2.0f);
        first = toPixel(foot - half_chord);
        second = toPixel(foot + half_chord);

        return cv::clipLine(cv::Rect(cv::Point(0, 0), _size), first, second);
    }

    static cv::Point2f lineDirection(float theta)
    {
        return cv::Point2f(std::cos(theta), std::sin(theta));
    }

    static cv::Point2f lineNormal(float theta)
    {
        return cv::Point2f(-std::sin(theta), std::cos(theta));
    }

    static float chordLength(float rho)
    {
        return 2.0f * std::sqrt(std::max(0.0f, 1.0f - rho * rho));
    }

    bool visibleInterval(float theta, float rho, float& t_min, float& t_max) const
    {
        const cv::Point2f point = lineNormal(theta) * rho;
        const cv::Point2f direction = lineDirection(theta);
        t_min = -std::numeric_limits<float>::infinity();
        t_max = std::numeric_limits<float>::infinity();

        const auto clipAxis = [&t_min, &t_max](float point_axis, float direction_axis, float limit)
        {
            if (std::abs(direction_axis) < 1e-9f)
            {
                return std::abs(point_axis) <= limit;
            }

            const float first = (-limit - point_axis) / direction_axis;
            const float second = (limit - point_axis) / direction_axis;
            t_min = std::max(t_min, std::min(first, second));
            t_max = std::min(t_max, std::max(first, second));
            return t_max > t_min;
        };

        return clipAxis(point.x, direction.x, _half_extent.x) &&
            clipAxis(point.y, direction.y, _half_extent.y) &&
            t_max > t_min;
    }

    static void alignLine(float reference_theta, float& theta, float& rho)
    {
        while (theta - reference_theta > std::numbers::pi_v<float> / 2.0f)
        {
            theta -= std::numbers::pi_v<float>;
            rho = -rho;
        }

        while (theta - reference_theta < -std::numbers::pi_v<float> / 2.0f)
        {
            theta += std::numbers::pi_v<float>;
            rho = -rho;
        }
    }

    static void normalizeLine(float& theta, float& rho)
    {
        while (theta <= -std::numbers::pi_v<float> / 2.0f)
        {
            theta += std::numbers::pi_v<float>;
            rho = -rho;
        }

        while (theta > std::numbers::pi_v<float> / 2.0f)
        {
            theta -= std::numbers::pi_v<float>;
            rho = -rho;
        }
    }

private:
    cv::Size _size;
    cv::Point2f _center;
    float _scale = 1.0f;
    cv::Point2f _half_extent;
};

}
