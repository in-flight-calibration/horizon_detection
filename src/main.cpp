#include <horizon_detection.hpp>

#include <memory>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<horizon::HorizonDetection>();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);

    for (const rclcpp::Node::SharedPtr& child : node->children())
    {
        executor.add_node(child);
    }

    executor.spin();

    rclcpp::shutdown();
    return 0;
}