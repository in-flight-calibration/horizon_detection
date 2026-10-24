#include <horizon_detection/horizon_detection.hpp>

#include <memory>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<horizon::HorizonDetection>();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node->receiver());
    executor.spin();

    rclcpp::shutdown();
    return 0;
}