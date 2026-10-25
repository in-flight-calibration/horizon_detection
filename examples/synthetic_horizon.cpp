#include <detector/detector_factory.hpp>
#include <horizon.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{

static constexpr std::string_view detector_name = "alpha";
constexpr float rate_hz = 30.0f;
constexpr int width = 1280;
constexpr int height = 720;

const std::function<float(float)> theta = [](float time_seconds)
{
    return 0.5f * std::sin(0.3f * time_seconds);
};

const std::function<float(float)> rho = [](float time_seconds)
{
    return 0.3f * std::sin(0.5f * time_seconds);
};

horizon::Horizon generate(float time_seconds)
{
    horizon::Horizon result;
    result.theta = theta(time_seconds);
    result.rho = rho(time_seconds);
    return result;
}

void drawRunway(
    cv::Mat& image,
    const horizon::Horizon& horizon,
    float time_seconds,
    const horizon::HorizonUtils& utils)
{
    const cv::Point2f direction = horizon::HorizonUtils::lineDirection(horizon.theta);
    const cv::Point2f normal = horizon::HorizonUtils::lineNormal(horizon.theta);
    const cv::Point2f vanishing_point =
        normal * horizon.rho + direction * (0.3f * std::sin(0.2f * time_seconds));
    const float ground_angle = -std::numbers::pi_v<float> / 2.0f;
    const float runway_angle = 25.0f * std::numbers::pi_v<float> / 180.0f;
    const cv::Point2f left_direction(
        std::cos(horizon.theta + ground_angle - runway_angle),
        std::sin(horizon.theta + ground_angle - runway_angle));
    const cv::Point2f right_direction(
        std::cos(horizon.theta + ground_angle + runway_angle),
        std::sin(horizon.theta + ground_angle + runway_angle));
    const float runway_length = 3.0f;
    const cv::Point vertices[] = {
        utils.toPixel(vanishing_point),
        utils.toPixel(vanishing_point + left_direction * runway_length),
        utils.toPixel(vanishing_point + right_direction * runway_length)};
    cv::fillConvexPoly(image, vertices, 3, cv::Scalar(150, 150, 150), cv::LINE_AA);
}

struct TestResult
{
    float detection_percent;
    float theta_rmse_degrees;
    float rho_rmse;

    bool passes() const
    {
        return detection_percent >= 90.0f &&
            theta_rmse_degrees <= 1.0f &&
            rho_rmse <= 0.01f;
    }
};

TestResult calculateResult(
    uint64_t frames,
    uint64_t valid_detections,
    float theta_squared_error_sum,
    float rho_squared_error_sum)
{
    const float frame_count = static_cast<float>(frames);
    const float detection_percent = frame_count > 0.0f
        ? 100.0f * static_cast<float>(valid_detections) / frame_count
        : 0.0f;
    const float theta_rmse_degrees = valid_detections > 0
        ? std::sqrt(theta_squared_error_sum / static_cast<float>(valid_detections)) *
            180.0f / std::numbers::pi_v<float>
        : std::numeric_limits<float>::infinity();
    const float rho_rmse = valid_detections > 0
        ? std::sqrt(rho_squared_error_sum / static_cast<float>(valid_detections))
        : std::numeric_limits<float>::infinity();

    return {detection_percent, theta_rmse_degrees, rho_rmse};
}

class HorizonRenderer
{
public:
    explicit HorizonRenderer(cv::Size size)
        : _utils(size)
    {
        _u.create(size, CV_32F);
        _v.create(size, CV_32F);

        for (int y = 0; y < size.height; ++y)
        {
            float* u_row = _u.ptr<float>(y);
            float* v_row = _v.ptr<float>(y);

            for (int x = 0; x < size.width; ++x)
            {
                const cv::Point2f point =
                    _utils.toNormalized(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
                u_row[x] = point.x;
                v_row[x] = point.y;
            }
        }
    }

    void render(cv::Mat& image, const horizon::Horizon& horizon, float time_seconds)
    {
        cv::Mat distance;
        cv::Mat sky_mask;
        cv::addWeighted(
            _u, -std::sin(horizon.theta),
            _v, std::cos(horizon.theta),
            -horizon.rho, distance
        );
        cv::compare(distance, 0.0, sky_mask, cv::CMP_GT);
        image.setTo(cv::Scalar(0, 0, 0));
        image.setTo(cv::Scalar(255, 255, 255), sky_mask);
        drawRunway(image, horizon, time_seconds, _utils);
    }

private:
    horizon::HorizonUtils _utils;
    cv::Mat _u;
    cv::Mat _v;
};

}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    std::unique_ptr<horizon::Detector> detector = horizon::DetectorFactory::create(std::string(detector_name));
    const double duration_seconds = detector->declare_parameter<double>("duration", 0.0);

    if (!std::isfinite(duration_seconds) || duration_seconds < 0.0)
    {
        throw std::invalid_argument("duration must be a finite non-negative value");
    }

    cv::Mat image(height, width, CV_8UC3);
    HorizonRenderer renderer(image.size());
    rclcpp::WallRate wall_rate(static_cast<double>(rate_hz));

    uint64_t frames = 0;
    uint64_t window_frames = 0;
    uint64_t valid_detections = 0;
    uint64_t total_valid_detections = 0;
    float theta_squared_error_sum = 0.0f;
    float rho_squared_error_sum = 0.0f;
    float total_theta_squared_error_sum = 0.0f;
    float total_rho_squared_error_sum = 0.0f;
    float confidence_sum = 0.0f;

    while (rclcpp::ok() &&
        (duration_seconds == 0.0 || static_cast<double>(frames) / rate_hz < duration_seconds))
    {
        const double time_seconds = static_cast<double>(frames) / rate_hz;
        const horizon::Horizon truth = generate(static_cast<float>(time_seconds));
        renderer.render(image, truth, static_cast<float>(time_seconds));

        const horizon::Horizon estimate = detector->detect(
            rclcpp::Time(static_cast<int64_t>(time_seconds * 1e9)),
            image
        );
        ++window_frames;
        confidence_sum += estimate.confidence;

        if (estimate.confidence > 0.0f)
        {
            float aligned_theta = estimate.theta;
            float aligned_rho = estimate.rho;
            horizon::HorizonUtils::alignLine(truth.theta, aligned_theta, aligned_rho);
            const float theta_error = aligned_theta - truth.theta;
            const float rho_error = aligned_rho - truth.rho;
            theta_squared_error_sum += theta_error * theta_error;
            rho_squared_error_sum += rho_error * rho_error;
            ++valid_detections;
            total_theta_squared_error_sum += theta_error * theta_error;
            total_rho_squared_error_sum += rho_error * rho_error;
            ++total_valid_detections;
        }

        ++frames;
        if (frames % static_cast<uint64_t>(rate_hz) == 0)
        {
            const TestResult result = calculateResult(
                window_frames, valid_detections, theta_squared_error_sum, rho_squared_error_sum);
            const float average_confidence = confidence_sum / static_cast<float>(window_frames);

            RCLCPP_INFO(
                detector->get_logger(),
                "t=%.0fs RMSE theta=%.3f deg rho=%.4f valid=%.1f%% avg confidence=%.3f",
                static_cast<double>(frames) / rate_hz,
                static_cast<double>(result.theta_rmse_degrees),
                static_cast<double>(result.rho_rmse),
                static_cast<double>(result.detection_percent),
                static_cast<double>(average_confidence));

            window_frames = 0;
            valid_detections = 0;
            theta_squared_error_sum = 0.0f;
            rho_squared_error_sum = 0.0f;
            confidence_sum = 0.0f;
        }

        wall_rate.sleep();
    }

    rclcpp::shutdown();

    if (duration_seconds == 0.0)
    {
        return 0;
    }

    const TestResult result = calculateResult(
        frames,
        total_valid_detections,
        total_theta_squared_error_sum,
        total_rho_squared_error_sum);

    RCLCPP_INFO(
        detector->get_logger(),
        "Test result: frames=%llu valid=%.1f%% RMSE theta=%.3f deg rho=%.4f",
        static_cast<unsigned long long>(frames),
        static_cast<double>(result.detection_percent),
        static_cast<double>(result.theta_rmse_degrees),
        static_cast<double>(result.rho_rmse));

    return result.passes() ? 0 : 1;
}
