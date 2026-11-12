#pragma once

#include <horizon.hpp>
#include <detector/alpha/line_grouping.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace horizon::hough
{

struct HorizonEstimatorParams
{
    float minimal_init_score = 0.33f;
    float minimal_update_score = 0.6f;
    int timeout_ms = 100;
    float max_theta_rate = 90.0f;
    float max_rho_rate = 1.0f;
    float confidance_filter_tau = 0.5f;
};

class HorizonEstimator
{
public:
    explicit HorizonEstimator(const HorizonEstimatorParams& params)
        : _params(params)
    {
        if (!std::isfinite(_params.minimal_init_score) ||
            _params.minimal_init_score < 0.0f ||
            _params.minimal_init_score > 1.0f ||
            !std::isfinite(_params.minimal_update_score) ||
            _params.minimal_update_score < 0.0f ||
            _params.minimal_update_score > 1.0f ||
            _params.timeout_ms <= 0 ||
            !std::isfinite(_params.max_theta_rate) ||
            _params.max_theta_rate <= 0.0f ||
            !std::isfinite(_params.max_rho_rate) ||
            _params.max_rho_rate <= 0.0f ||
            !std::isfinite(_params.confidance_filter_tau) ||
            _params.confidance_filter_tau <= 0.0f)
        {
            throw std::invalid_argument("Invalid horizon estimator parameter");
        }
    }

    Horizon estimate(int64_t timestamp_ns, const std::vector<LineGroup>& groups)
    {
        if (groups.empty())
        {
            return Detector::INVALID_HORIZON;
        }

        const float dt = checkTimestamp(timestamp_ns);

        const std::vector<float> scores = score(dt, groups);
        const std::size_t best_index = static_cast<std::size_t>(
            std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
        const LineGroup best = groups[best_index];
        
        const float best_score = scores[best_index];
        const float minimal_score = _last_horizon ? _params.minimal_update_score : _params.minimal_init_score;

        if (best_score < minimal_score)
        {
            return Detector::INVALID_HORIZON;
        }

        const float alpha = _last_horizon ? 1.0f - std::exp(-dt / _params.confidance_filter_tau) : 1.0f;
        _confidance += alpha * (best_score - _confidance);
        return finalize(timestamp_ns, best);
    }

    static cv::Mat getDebugView(
        const cv::Mat& image,
        const Horizon& horizon,
        const HorizonUtils& utils)
    {
        cv::Mat output = image.clone();

        if (horizon.confidence <= 0.0f)
        {
            return output;
        }

        cv::Point first;
        cv::Point second;

        if (utils.endpoints(horizon, first, second))
        {
            cv::line(output, first, second, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }

        return output;
    }

private:
    HorizonEstimatorParams _params;
    std::optional<Horizon> _last_horizon;
    int64_t _last_horizon_update_ns = 0;
    float _confidance = 0.0f;

    float checkTimestamp(int64_t timestamp_ns)
    {
        if (!_last_horizon)
        {
            return 0.0f;
        }

        const float dt = static_cast<float>(timestamp_ns - _last_horizon_update_ns) * 1e-9f;

        if (dt < 0.0f || dt * 1000.0f > static_cast<float>(_params.timeout_ms))
        {
            _last_horizon = std::nullopt;
            _last_horizon_update_ns = 0;
            _confidance = 0.0f;
            return 0.0f;
        }

        return dt;
    }

    std::vector<float> score(float dt, const std::vector<LineGroup>& groups) const
    {
        const std::vector<float> coverage = scoreCoverage(groups);
        const std::vector<float> top = scoreTop(groups);
        const std::vector<float> coherence = scoreCoherence(dt, groups);
        std::vector<float> scores(groups.size());

        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            scores[i] = (coverage[i] + top[i] + coherence[i]) / 3.0f;
        }

        return scores;
    }

    static std::vector<float> scoreCoverage(const std::vector<LineGroup>& groups)
    {
        std::vector<float> scores(groups.size());

        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            scores[i] = std::clamp(groups[i].coverage, 0.0f, 1.0f);
        }

        return scores;
    }

    static std::vector<float> scoreTop(const std::vector<LineGroup>& groups)
    {
        std::vector<float> scores(groups.size(), 1.0f);

        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            const cv::Point2f normal = HorizonUtils::lineNormal(groups[i].theta);
            float below_length = 0.0f;
            float total_length = 0.0f;

            for (std::size_t j = 0; j < groups.size(); ++j)
            {
                if (j == i)
                {
                    continue;
                }

                for (const Segment& segment : groups[j].segments)
                {
                    const float first = normal.dot(segment.first) - groups[i].rho;
                    const float second = normal.dot(segment.second) - groups[i].rho;
                    float below_fraction = 0.0f;

                    if (first < 0.0f && second < 0.0f)
                    {
                        below_fraction = 1.0f;
                    }
                    else if (first < 0.0f)
                    {
                        below_fraction = -first / (second - first);
                    }
                    else if (second < 0.0f)
                    {
                        below_fraction = -second / (first - second);
                    }

                    total_length += segment.length;
                    below_length += segment.length * below_fraction;
                }
            }

            if (total_length > 0.0f)
            {
                scores[i] = std::clamp(below_length / total_length, 0.0f, 1.0f);
            }
        }

        return scores;
    }

    std::vector<float> scoreCoherence(float dt, const std::vector<LineGroup>& groups) const
    {
        std::vector<float> scores(groups.size(), 0.0f);

        if (!_last_horizon)
        {
            return scores;
        }

        const float step = std::max(dt, 0.05f);
        const float max_theta_delta = _params.max_theta_rate * std::numbers::pi_v<float> / 180.0f * step;
        const float max_rho_delta = _params.max_rho_rate * step;

        for (std::size_t i = 0; i < groups.size(); ++i)
        {
            float theta = groups[i].theta;
            float rho = groups[i].rho;
            HorizonUtils::alignLine(_last_horizon->theta, theta, rho);
            const float distance = std::max(
                std::abs(theta - _last_horizon->theta) / max_theta_delta,
                std::abs(rho - _last_horizon->rho) / max_rho_delta);
            scores[i] = std::max(0.0f, 1.0f - distance);
        }

        return scores;
    }

    Horizon finalize(int64_t timestamp_ns, const LineGroup& group)
    {
        Horizon horizon = makeHorizon(group);
        horizon.confidence = _confidance;
        _last_horizon = horizon;
        _last_horizon_update_ns = timestamp_ns;
        return horizon;
    }

    static Horizon makeHorizon(const LineGroup& group)
    {
        Horizon horizon;
        horizon.theta = group.theta;
        horizon.rho = std::clamp(group.rho, -1.0f, 1.0f);

        const float segment_count = static_cast<float>(group.segments.size());
        const float length = std::accumulate(
            group.segments.begin(), group.segments.end(), 0.0f,
            [](float sum, const Segment& segment)
            {
                return sum + segment.length;
            });
        const float root_count = std::sqrt(std::max(segment_count, 1.0f));
        const float projection_spread = std::max(length / std::sqrt(12.0f), 1e-6f);

        horizon.sigma_rho = group.residual / root_count;
        horizon.sigma_theta = group.residual / (root_count * projection_spread);

        return horizon;
    }
};

}
