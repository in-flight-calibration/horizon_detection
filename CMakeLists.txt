cmake_minimum_required(VERSION 3.20)
project(horizon_detection)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ifc_msgs REQUIRED)
find_package(ifc_tools REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(rclcpp REQUIRED)

function(add_horizon_executable name source)
    add_executable(${name} ${source})

    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(${name} PRIVATE
        ifc_tools::ifc_tools
        ifc_msgs::ifc_msgs__rosidl_typesupport_cpp
        rclcpp::rclcpp
        ${OpenCV_LIBS}
    )

    install(TARGETS ${name}
        DESTINATION lib/${PROJECT_NAME}
    )
endfunction()

add_horizon_executable(horizon_detection src/main.cpp)
add_horizon_executable(synthetic_horizon examples/synthetic_horizon.cpp)

ament_export_dependencies(ifc_msgs ifc_tools OpenCV rclcpp)

ament_package()