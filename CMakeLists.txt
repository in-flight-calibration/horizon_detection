cmake_minimum_required(VERSION 3.20)
project(horizon_detection)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

add_executable(horizon_detection_node src/main.cpp)

install(TARGETS horizon_detection_node
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()