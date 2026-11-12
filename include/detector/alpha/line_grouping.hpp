#pragma once

#include <horizon.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace horizon::hough
{

struct LineGroupingParams
{
    float distance_tolerance = 0.02f;
    float minimal_coverage = 0.5f;
    int slots = 3;
};

struct Segment
{
    cv::Point2f first;
    cv::Point2f second;
    float length = 0.0f;
};

struct LineGroup
{
    float theta = 0.0f;
    float rho = 0.0f;

    float coverage = 0.0f;
    float residual = 0.0f;

    std::vector<Segment> segments;
};

class LineGrouping
{
public:
    explicit LineGrouping(const LineGroupingParams& params)
        : _params(params)
    {
        if (!std::isfinite(_params.distance_tolerance) ||
            _params.distance_tolerance <= 0.0f ||
            !std::isfinite(_params.minimal_coverage) ||
            _params.minimal_coverage < 0.0f ||
            _params.slots < 1)
        {
            throw std::invalid_argument("Invalid line grouping parameter");
        }
    }

    bool group(const std::vector<cv::Vec4i>& lines, const HorizonUtils& utils, std::vector<LineGroup>& result) const
    {
        std::vector<Segment> segments = makeSegments(lines, utils);
        sortSegments(segments);
        std::vector<LineGroup> groups = makeGroups(segments);

        for (LineGroup& group : groups)
        {
            evaluateGroup(group, utils);
        }

        std::erase_if(
            groups,
            [this](const LineGroup& group)
            {
                return group.coverage < _params.minimal_coverage;
            });

        sortGroups(groups);
        groups.resize(std::min(groups.size(), static_cast<std::size_t>(_params.slots)));
        result = std::move(groups);

        return true;
    }

    static cv::Mat getDebugView(
        cv::Size size,
        const std::vector<LineGroup>& groups,
        const HorizonUtils& utils)
    {
        cv::Mat output = cv::Mat::zeros(size, CV_8UC3);
        const int color_count = std::max(1, static_cast<int>(groups.size()));
        const std::vector<cv::Vec3b> colors = generate_colors(color_count);

        for (std::size_t index = 0; index < groups.size(); ++index)
        {
            const LineGroup& group = groups[index];
            const cv::Vec3b& color = colors[index];
            const cv::Scalar bgr_color(color[0], color[1], color[2]);

            for (const Segment& segment : group.segments)
            {
                cv::line(output, utils.toPixel(segment.first), utils.toPixel(segment.second), bgr_color, 1, cv::LINE_AA);
            }

            float t_min = 0.0f;
            float t_max = 0.0f;

            if (!utils.visibleInterval(group.theta, group.rho, t_min, t_max))
            {
                continue;
            }

            const cv::Point2f point = HorizonUtils::lineNormal(group.theta) * group.rho;
            const cv::Point2f direction = HorizonUtils::lineDirection(group.theta);
            cv::line(
                output,
                utils.toPixel(point + direction * t_min),
                utils.toPixel(point + direction * t_max),
                bgr_color,
                1,
                cv::LINE_AA);

        }

        return output;
    }

private:
    LineGroupingParams _params;

    static std::vector<Segment> makeSegments(const std::vector<cv::Vec4i>& lines, const HorizonUtils& utils)
    {
        std::vector<Segment> segments;

        for (const cv::Vec4i& line : lines)
        {
            const cv::Point2f first = utils.toNormalized(cv::Point2f(static_cast<float>(line[0]), static_cast<float>(line[1])));
            const cv::Point2f second = utils.toNormalized(cv::Point2f(static_cast<float>(line[2]), static_cast<float>(line[3])));
            const float length = static_cast<float>(cv::norm(second - first));

            if (length > 0.0f)
            {
                segments.push_back({first, second, length});
            }
        }

        return segments;
    }

    static void sortSegments(std::vector<Segment>& segments)
    {
        std::sort(
            segments.begin(), segments.end(),
            [](const Segment& lhs, const Segment& rhs)
            {
                return lhs.length > rhs.length;
            });
    }

    static float distance(const cv::Vec4f& line, const Segment& segment)
    {
        const cv::Point2f direction(line[0], line[1]);
        const cv::Point2f origin(line[2], line[3]);
        return std::max(
            std::abs(direction.cross(segment.first - origin)),
            std::abs(direction.cross(segment.second - origin)));
    }

    static cv::Vec4f fit(const LineGroup& group)
    {
        std::vector<cv::Point2f> points;

        for (const Segment& segment : group.segments)
        {
            points.insert(points.end(), {segment.first, segment.second});
        }

        cv::Vec4f line;
        cv::fitLine(points, line, cv::DIST_L2, 0, 0.01, 0.01);
        return line;
    }

    std::vector<LineGroup> makeGroups(const std::vector<Segment>& segments) const
    {
        std::vector<LineGroup> groups;
        std::vector<cv::Vec4f> lines;

        for (const Segment& segment : segments)
        {
            std::size_t best = groups.size();
            float best_distance = _params.distance_tolerance;

            for (std::size_t index = 0; index < groups.size(); ++index)
            {
                const float worst = distance(lines[index], segment);

                if (worst < best_distance)
                {
                    best = index;
                    best_distance = worst;
                }
            }

            if (best == groups.size())
            {
                groups.emplace_back();
                lines.emplace_back();
            }

            groups[best].segments.push_back(segment);
            lines[best] = fit(groups[best]);
        }

        return groups;
    }

    static float measureCoverage(const LineGroup& group, const HorizonUtils& utils)
    {
        float t_min = 0.0f;
        float t_max = 0.0f;

        if (!utils.visibleInterval(group.theta, group.rho, t_min, t_max))
        {
            return 0.0f;
        }

        const cv::Point2f direction = HorizonUtils::lineDirection(group.theta);
        std::vector<std::pair<float, float>> intervals;

        for (const Segment& segment : group.segments)
        {
            const float first = direction.dot(segment.first);
            const float second = direction.dot(segment.second);
            const float start = std::max(t_min, std::min(first, second));
            const float end = std::min(t_max, std::max(first, second));

            if (end > start)
            {
                intervals.emplace_back(start, end);
            }
        }

        std::sort(intervals.begin(), intervals.end());

        float covered = 0.0f;
        float reach = t_min;

        for (const auto& [start, end] : intervals)
        {
            covered += std::max(0.0f, end - std::max(start, reach));
            reach = std::max(reach, end);
        }

        return std::clamp(covered / (t_max - t_min), 0.0f, 1.0f);
    }

    static void evaluateGroup(LineGroup& group, const HorizonUtils& utils)
    {
        const cv::Vec4f line = fit(group);
        group.theta = std::atan2(line[1], line[0]);
        group.rho = HorizonUtils::lineNormal(group.theta).dot(cv::Point2f(line[2], line[3]));
        HorizonUtils::normalizeLine(group.theta, group.rho);

        const double squared = std::accumulate(
            group.segments.begin(), group.segments.end(), 0.0,
            [&line](double total, const Segment& segment)
            {
                const double residual = distance(line, segment);
                return total + residual * residual;
            });
        group.residual = static_cast<float>(std::sqrt(squared / static_cast<double>(group.segments.size())));
        group.coverage = measureCoverage(group, utils);
    }

    void sortGroups(std::vector<LineGroup>& groups) const
    {
        std::sort(
            groups.begin(), groups.end(),
            [](const LineGroup& lhs, const LineGroup& rhs)
            {
                return lhs.coverage > rhs.coverage;
            });
    }

    static std::vector<cv::Vec3b> generate_colors(int count)
    {
        cv::Mat indices(1, count, CV_8UC1);

        for (int index = 0; index < count; ++index)
        {
            indices.at<uchar>(0, index) = static_cast<uchar>(
                static_cast<std::size_t>(index) * 256 / static_cast<std::size_t>(count));
        }

        cv::Mat colors;
        cv::applyColorMap(indices, colors, cv::COLORMAP_TURBO);

        std::vector<cv::Vec3b> result(static_cast<std::size_t>(count));

        for (int index = 0; index < count; ++index)
        {
            result[static_cast<std::size_t>(index)] = colors.at<cv::Vec3b>(0, index);
        }

        return result;
    }
};

}
