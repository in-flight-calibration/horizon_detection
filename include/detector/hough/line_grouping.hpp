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
    float angle_tolerance_degrees = 2.0f;
    float rho_tolerance = 0.5f;
    int slots = 5;
};

struct Segment
{
    cv::Point2f first;
    cv::Point2f second;
    float length = 0.0f;
    float theta = 0.0f;
    float rho = 0.0f;
};

struct LineGroup
{
    std::vector<std::size_t> segment_indices;
    float theta = 0.0f;
    float rho = 0.0f;
    float total_length = 0.0f;
    float covered_length = 0.0f;
    float visible_length = 0.0f;
    float coverage = 0.0f;
    float residual = 0.0f;
};

struct LineGroups
{
    std::vector<Segment> segments;
    std::vector<LineGroup> groups;
};

class LineGrouping
{
public:
    explicit LineGrouping(const LineGroupingParams& params)
        : _params(params),
          _angle_tolerance(params.angle_tolerance_degrees * std::numbers::pi_v<float> / 180.0f)
    {
        if (!std::isfinite(_angle_tolerance) ||
            !std::isfinite(_params.rho_tolerance) ||
            _angle_tolerance <= 0.0f ||
            _angle_tolerance >= std::numbers::pi_v<float> / 2.0f ||
            _params.rho_tolerance <= 0.0f ||
            _params.slots < 1)
        {
            throw std::invalid_argument("Invalid line grouping parameter");
        }
    }

    bool group(const std::vector<cv::Vec4i>& lines, const HorizonUtils& utils, LineGroups& result) const
    {
        result = LineGroups();
        result.segments = makeSegments(lines, utils);
        const std::vector<Cluster> clusters = cluster(result.segments);
        result.groups.reserve(clusters.size());

        for (const Cluster& cluster : clusters)
        {
            LineGroup fitted;

            if (fit(cluster, result.segments, fitted))
            {
                measureCoverage(fitted, result.segments, utils);
                result.groups.push_back(std::move(fitted));
            }
        }

        std::sort(
            result.groups.begin(), result.groups.end(),
            [](const LineGroup& lhs, const LineGroup& rhs)
            {
                return lhs.coverage > rhs.coverage;
            });

        return true;
    }

    static cv::Mat get_debug_view(
        cv::Size size,
        const LineGroups& groups,
        const HorizonUtils& utils)
    {
        cv::Mat output = cv::Mat::zeros(size, CV_8UC3);
        const int color_count = std::max(1, static_cast<int>(groups.groups.size()));
        const std::vector<cv::Vec3b> colors = generate_colors(color_count);

        for (std::size_t index = 0; index < groups.groups.size(); ++index)
        {
            const LineGroup& group = groups.groups[index];
            const cv::Vec3b& color = colors[index];
            const cv::Scalar bgr_color(color[0], color[1], color[2]);

            for (const std::size_t segment_index : group.segment_indices)
            {
                const Segment& segment = groups.segments[segment_index];
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
                3,
                cv::LINE_AA);

        }

        return output;
    }

private:
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

    struct Cluster
    {
        std::vector<std::size_t> segment_indices;
        float weight = 0.0f;
        float theta_sum = 0.0f;
        float rho_sum = 0.0f;
        float reference_theta = 0.0f;
        float reference_rho = 0.0f;
    };

    LineGroupingParams _params;
    float _angle_tolerance;

    static std::vector<Segment> makeSegments(const std::vector<cv::Vec4i>& lines, const HorizonUtils& utils)
    {
        std::vector<Segment> segments;
        segments.reserve(lines.size());

        for (const cv::Vec4i& line : lines)
        {
            Segment segment;
            segment.first = utils.toNormalized(cv::Point2f(static_cast<float>(line[0]), static_cast<float>(line[1])));
            segment.second = utils.toNormalized(cv::Point2f(static_cast<float>(line[2]), static_cast<float>(line[3])));
            const cv::Point2f delta = segment.second - segment.first;
            segment.length = std::hypot(delta.x, delta.y);

            if (segment.length <= 0.0f)
            {
                continue;
            }

            segment.theta = std::atan2(delta.y, delta.x);
            HorizonUtils::normalizeLine(segment.theta, segment.rho);
            const cv::Point2f midpoint = (segment.first + segment.second) * 0.5f;
            segment.rho = HorizonUtils::lineNormal(segment.theta).dot(midpoint);
            segments.push_back(segment);
        }

        return segments;
    }

    std::vector<Cluster> cluster(const std::vector<Segment>& segments) const
    {
        std::vector<std::size_t> order(segments.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(
            order.begin(), order.end(),
            [&segments](std::size_t lhs, std::size_t rhs)
            {
                return segments[lhs].length > segments[rhs].length;
            });

        std::vector<Cluster> clusters;
        clusters.reserve(static_cast<std::size_t>(_params.slots));

        for (const std::size_t index : order)
        {
            const Segment& segment = segments[index];
            std::size_t best_cluster = clusters.size();
            float best_distance = std::numeric_limits<float>::infinity();
            float best_theta = segment.theta;
            float best_rho = segment.rho;

            for (std::size_t cluster_index = 0; cluster_index < clusters.size(); ++cluster_index)
            {
                const Cluster& candidate = clusters[cluster_index];
                float theta = segment.theta;
                float rho = segment.rho;
                HorizonUtils::alignLine(candidate.reference_theta, theta, rho);
                const float angle_difference = std::abs(theta - candidate.reference_theta);
                const float rho_difference = std::abs(rho - candidate.reference_rho);

                if (angle_difference >= _angle_tolerance || rho_difference >= _params.rho_tolerance)
                {
                    continue;
                }

                const float distance = angle_difference / _angle_tolerance + rho_difference / _params.rho_tolerance;

                if (distance < best_distance)
                {
                    best_cluster = cluster_index;
                    best_distance = distance;
                    best_theta = theta;
                    best_rho = rho;
                }
            }

            if (best_cluster == clusters.size())
            {
                if (clusters.size() >= static_cast<std::size_t>(_params.slots))
                {
                    continue;
                }

                clusters.emplace_back();
                best_cluster = clusters.size() - 1;
                best_theta = segment.theta;
                best_rho = segment.rho;
                clusters[best_cluster].reference_theta = best_theta;
                clusters[best_cluster].reference_rho = best_rho;
            }

            Cluster& selected = clusters[best_cluster];
            selected.segment_indices.push_back(index);
            selected.weight += segment.length;
            selected.theta_sum += segment.length * best_theta;
            selected.rho_sum += segment.length * best_rho;
            selected.reference_theta = selected.theta_sum / selected.weight;
            selected.reference_rho = selected.rho_sum / selected.weight;
        }

        return clusters;
    }

    static bool fit(const Cluster& cluster, const std::vector<Segment>& segments, LineGroup& group)
    {
        double weight_sum = 0.0;
        double center_u_sum = 0.0;
        double center_v_sum = 0.0;

        for (const std::size_t index : cluster.segment_indices)
        {
            const Segment& segment = segments[index];
            const double weight = static_cast<double>(segment.length);
            const double midpoint_u = (static_cast<double>(segment.first.x) + segment.second.x) / 2.0;
            const double midpoint_v = (static_cast<double>(segment.first.y) + segment.second.y) / 2.0;
            weight_sum += weight;
            center_u_sum += weight * midpoint_u;
            center_v_sum += weight * midpoint_v;
        }

        if (weight_sum <= 0.0)
        {
            return false;
        }

        const double center_u = center_u_sum / weight_sum;
        const double center_v = center_v_sum / weight_sum;
        double suu = 0.0;
        double svv = 0.0;
        double suv = 0.0;

        for (const std::size_t index : cluster.segment_indices)
        {
            const Segment& segment = segments[index];
            const double weight = static_cast<double>(segment.length);
            const double delta_u = static_cast<double>(segment.second.x) - segment.first.x;
            const double delta_v = static_cast<double>(segment.second.y) - segment.first.y;
            const double midpoint_u = (static_cast<double>(segment.first.x) + segment.second.x) / 2.0;
            const double midpoint_v = (static_cast<double>(segment.first.y) + segment.second.y) / 2.0;
            const double offset_u = midpoint_u - center_u;
            const double offset_v = midpoint_v - center_v;
            suu += weight * (offset_u * offset_u + delta_u * delta_u / 12.0);
            svv += weight * (offset_v * offset_v + delta_v * delta_v / 12.0);
            suv += weight * (offset_u * offset_v + delta_u * delta_v / 12.0);
        }

        float theta = static_cast<float>(0.5 * std::atan2(2.0 * suv, suu - svv));
        const cv::Point2f center(static_cast<float>(center_u), static_cast<float>(center_v));
        float rho = HorizonUtils::lineNormal(theta).dot(center);
        HorizonUtils::normalizeLine(theta, rho);

        const double eigenvalue_minimum =
            0.5 * (suu + svv) - std::sqrt(0.25 * (suu - svv) * (suu - svv) + suv * suv);

        group.segment_indices = cluster.segment_indices;
        group.theta = theta;
        group.rho = rho;
        group.total_length = static_cast<float>(weight_sum);
        group.residual = static_cast<float>(std::sqrt(std::max(0.0, eigenvalue_minimum) / weight_sum));
        return true;
    }

    static void measureCoverage(LineGroup& group, const std::vector<Segment>& segments, const HorizonUtils& utils)
    {
        float t_min = 0.0f;
        float t_max = 0.0f;

        if (!utils.visibleInterval(group.theta, group.rho, t_min, t_max))
        {
            group.covered_length = 0.0f;
            group.visible_length = 0.0f;
            group.coverage = 0.0f;
            return;
        }

        group.visible_length = t_max - t_min;
        const cv::Point2f direction = HorizonUtils::lineDirection(group.theta);
        std::vector<std::pair<float, float>> intervals;
        intervals.reserve(group.segment_indices.size());

        for (const std::size_t index : group.segment_indices)
        {
            const Segment& segment = segments[index];
            const float first = direction.dot(segment.first);
            const float second = direction.dot(segment.second);
            const float start = std::max(t_min, std::min(first, second));
            const float end = std::min(t_max, std::max(first, second));

            if (end > start)
            {
                intervals.emplace_back(start, end);
            }
        }

        std::sort(
            intervals.begin(), intervals.end(),
            [](const std::pair<float, float>& lhs, const std::pair<float, float>& rhs)
            {
                return lhs.first < rhs.first;
            });

        group.covered_length = 0.0f;

        if (!intervals.empty())
        {
            float current_start = intervals.front().first;
            float current_end = intervals.front().second;

            for (std::size_t index = 1; index < intervals.size(); ++index)
            {
                if (intervals[index].first <= current_end)
                {
                    current_end = std::max(current_end, intervals[index].second);
                }
                else
                {
                    group.covered_length += current_end - current_start;
                    current_start = intervals[index].first;
                    current_end = intervals[index].second;
                }
            }

            group.covered_length += current_end - current_start;
        }

        group.coverage = group.visible_length > 0.0f
            ? std::clamp(group.covered_length / group.visible_length, 0.0f, 1.0f)
            : 0.0f;
    }
};

}
