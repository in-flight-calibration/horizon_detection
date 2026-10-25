#pragma once

#include <horizon.hpp>
#include <detector/hough/line_grouping.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace horizon::hough
{

struct HorizonEstimatorParams
{
    float minimum_coverage = 0.1f;
    float minimum_score = 0.5f;
    float ambiguity_margin = 0.1f;
    float stale_seconds = 0.1f;
    float max_theta_rate_degrees_per_second = 90.0f;
    float max_rho_rate_per_second = 1.0f;
    float tau_rise_seconds = 0.5f;
    float minimum_confidence = 0.3f;
};

class HorizonEstimator
{
public:
    explicit HorizonEstimator(const HorizonEstimatorParams& params)
        : _params(params)
    {
        if (!std::isfinite(_params.minimum_coverage) ||
            _params.minimum_coverage < 0.0f ||
            _params.minimum_coverage > 1.0f ||
            !std::isfinite(_params.minimum_score) ||
            _params.minimum_score < 0.0f ||
            _params.minimum_score > 1.0f ||
            !std::isfinite(_params.ambiguity_margin) ||
            _params.ambiguity_margin < 0.0f ||
            _params.ambiguity_margin > 1.0f ||
            !std::isfinite(_params.stale_seconds) ||
            _params.stale_seconds <= 0.0f ||
            !std::isfinite(_params.max_theta_rate_degrees_per_second) ||
            _params.max_theta_rate_degrees_per_second <= 0.0f ||
            !std::isfinite(_params.max_rho_rate_per_second) ||
            _params.max_rho_rate_per_second <= 0.0f ||
            !std::isfinite(_params.tau_rise_seconds) ||
            _params.tau_rise_seconds <= 0.0f ||
            !std::isfinite(_params.minimum_confidence) ||
            _params.minimum_confidence < 0.0f ||
            _params.minimum_confidence > 1.0f)
        {
            throw std::invalid_argument("Invalid horizon estimator parameter");
        }
    }

    Horizon estimate(int64_t timestamp_ns, const LineGroups& groups)
    {
        const float frame_elapsed_seconds = _has_timestamp && timestamp_ns > _last_timestamp_ns
            ? static_cast<float>(timestamp_ns - _last_timestamp_ns) * 1e-9f
            : 0.0f;
        _last_timestamp_ns = timestamp_ns;
        _has_timestamp = true;

        const int64_t stale_ns = static_cast<int64_t>(_params.stale_seconds * 1e9f);
        const int64_t track_age_ns = timestamp_ns - _track_timestamp_ns;
        const bool rate_reference_valid = _has_track && track_age_ns >= 0;
        const bool track_fresh = rate_reference_valid && track_age_ns <= stale_ns;
        const float track_elapsed_seconds = rate_reference_valid
            ? std::max(static_cast<float>(track_age_ns) * 1e-9f, 0.05f)
            : 0.0f;
        const float max_theta_delta =
            _params.max_theta_rate_degrees_per_second * std::numbers::pi_v<float> / 180.0f * track_elapsed_seconds;
        const float max_rho_delta = _params.max_rho_rate_per_second * track_elapsed_seconds;

        std::vector<Candidate> candidates;
        candidates.reserve(groups.groups.size());

        for (std::size_t group_index = 0; group_index < groups.groups.size(); ++group_index)
        {
            const LineGroup& group = groups.groups[group_index];

            if (group.coverage < _params.minimum_coverage)
            {
                continue;
            }

            Candidate candidate;
            candidate.group_index = group_index;
            candidate.mu_coverage = std::clamp(group.coverage, 0.0f, 1.0f);
            candidate.mu_top = topmost(group_index, groups);

            if (rate_reference_valid)
            {
                float theta = group.theta;
                float rho = group.rho;
                HorizonUtils::alignLine(_track.theta, theta, rho);
                const float theta_delta = std::abs(theta - _track.theta);
                const float rho_delta = std::abs(rho - _track.rho);
                candidate.distance = std::max(theta_delta / max_theta_delta, rho_delta / max_rho_delta);

                if (candidate.distance > 1.0f)
                {
                    continue;
                }

                if (track_fresh)
                {
                    candidate.mu_previous = std::max(0.0f, 1.0f - candidate.distance);
                    candidate.has_previous = true;
                }
            }

            const float criterion_count = candidate.has_previous ? 3.0f : 2.0f;
            candidate.score = (candidate.mu_coverage + candidate.mu_top + candidate.mu_previous) / criterion_count;
            candidates.push_back(candidate);
        }

        if (candidates.empty())
        {
            clearStaleTrack(timestamp_ns, stale_ns);
            return trackedOutput(track_fresh);
        }

        const auto best_iterator = std::max_element(
            candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs)
            {
                return lhs.score < rhs.score;
            });
        const Candidate& best = *best_iterator;
        float second_best_score = -1.0f;

        for (const Candidate& candidate : candidates)
        {
            if (&candidate != &best)
            {
                second_best_score = std::max(second_best_score, candidate.score);
            }
        }

        const Horizon candidate_horizon = makeHorizon(groups.groups[best.group_index]);

        if (best.score < _params.minimum_score ||
            (second_best_score >= 0.0f && best.score - second_best_score < _params.ambiguity_margin))
        {
            clearStaleTrack(timestamp_ns, stale_ns);
            return trackedOutput(track_fresh);
        }

        const Candidate* closest = nullptr;

        if (track_fresh)
        {
            for (const Candidate& candidate : candidates)
            {
                if (!candidate.has_previous)
                {
                    continue;
                }

                if (closest == nullptr || candidate.distance < closest->distance)
                {
                    closest = &candidate;
                }
            }
        }

        const bool continuing = closest != nullptr &&
            closest->group_index == best.group_index &&
            best.distance <= 1.0f;

        if (!continuing)
        {
            _confidence = 0.0f;
        }

        const float confidence_alpha = 1.0f - std::exp(-frame_elapsed_seconds / _params.tau_rise_seconds);
        _confidence += confidence_alpha * (best.score - _confidence);
        _track = candidate_horizon;
        _track.confidence = _confidence;
        _track_timestamp_ns = timestamp_ns;
        _has_track = true;

        if (_confidence < _params.minimum_confidence)
        {
            return {};
        }

        Horizon result = candidate_horizon;
        result.confidence = _confidence;
        return result;
    }

    static cv::Mat get_debug_view(
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
    struct Candidate
    {
        std::size_t group_index = 0;
        float mu_coverage = 0.0f;
        float mu_top = 0.0f;
        float mu_previous = 0.0f;
        float distance = std::numeric_limits<float>::infinity();
        float score = 0.0f;
        bool has_previous = false;
    };

    HorizonEstimatorParams _params;
    Horizon _track;
    int64_t _track_timestamp_ns = 0;
    int64_t _last_timestamp_ns = 0;
    float _confidence = 0.0f;
    bool _has_track = false;
    bool _has_timestamp = false;

    static float topmost(std::size_t candidate_index, const LineGroups& groups)
    {
        const LineGroup& candidate_group = groups.groups[candidate_index];
        const cv::Point2f normal = HorizonUtils::lineNormal(candidate_group.theta);
        float below_length = 0.0f;
        float total_length = 0.0f;

        for (std::size_t group_index = 0; group_index < groups.groups.size(); ++group_index)
        {
            if (group_index == candidate_index)
            {
                continue;
            }

            for (const std::size_t segment_index : groups.groups[group_index].segment_indices)
            {
                if (segment_index >= groups.segments.size())
                {
                    continue;
                }

                const Segment& segment = groups.segments[segment_index];
                const float first_distance = normal.dot(segment.first) - candidate_group.rho;
                const float second_distance = normal.dot(segment.second) - candidate_group.rho;
                float below_fraction = 0.0f;

                if (first_distance < 0.0f && second_distance < 0.0f)
                {
                    below_fraction = 1.0f;
                }
                else if (first_distance < 0.0f && second_distance >= 0.0f)
                {
                    below_fraction = -first_distance / (second_distance - first_distance);
                }
                else if (first_distance >= 0.0f && second_distance < 0.0f)
                {
                    below_fraction = -second_distance / (first_distance - second_distance);
                }

                total_length += segment.length;
                below_length += segment.length * below_fraction;
            }
        }

        return total_length > 0.0f
            ? std::clamp(below_length / total_length, 0.0f, 1.0f)
            : 1.0f;
    }

    static Horizon makeHorizon(const LineGroup& group)
    {
        Horizon horizon;
        const float segment_count = static_cast<float>(group.segment_indices.size());
        const float root_count = std::sqrt(std::max(segment_count, 1.0f));
        const float projection_spread = std::max(group.covered_length / std::sqrt(12.0f), 1e-6f);

        horizon.theta = group.theta;
        horizon.rho = std::clamp(group.rho, -1.0f, 1.0f);
        horizon.length = group.covered_length;
        horizon.sigma_rho = group.residual / root_count;
        horizon.sigma_theta = group.residual / (root_count * projection_spread);
        return horizon;
    }

    void clearStaleTrack(int64_t timestamp_ns, int64_t stale_ns)
    {
        if (_has_track && timestamp_ns - _track_timestamp_ns > stale_ns)
        {
            _confidence = 0.0f;
            _track.confidence = 0.0f;
        }
    }

    Horizon trackedOutput(bool track_fresh) const
    {
        if (track_fresh && _confidence >= _params.minimum_confidence)
        {
            Horizon result = _track;
            result.confidence = _confidence;
            return result;
        }

        return {};
    }
};

}
