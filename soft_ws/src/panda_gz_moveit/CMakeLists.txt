cmake_minimum_required(VERSION 3.8)
project(panda_gz_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY
  launch
  urdf
  meshes
  config
  worlds
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/color_cube_detector.py
  scripts/camera_tf_publisher.py
  scripts/pick_place_controller.py
  scripts/camera_debug_visualizer.py
  scripts/simple_moveit_test.py
  DESTINATION lib/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()