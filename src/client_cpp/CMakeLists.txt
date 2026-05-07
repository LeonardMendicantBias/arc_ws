cmake_minimum_required(VERSION 3.8)
project(client_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(arc_interfaces REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Locate LibTorch (used for pre/post-processing and argmax)
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

# TensorRT is installed system-wide by JetPack
find_library(NVINFER_LIB nvinfer REQUIRED
  PATHS /usr/lib/aarch64-linux-gnu)
find_path(NVINFER_INCLUDE_DIR NvInfer.h REQUIRED
  PATHS /usr/include/aarch64-linux-gnu)

add_executable(codeword_publisher src/codeword_publisher.cpp)
target_include_directories(codeword_publisher PRIVATE
  include
  ${NVINFER_INCLUDE_DIR}
)
target_link_libraries(codeword_publisher
  "${TORCH_LIBRARIES}"
  ${NVINFER_LIB}
  CUDA::cudart
)
ament_target_dependencies(codeword_publisher
  rclcpp
  sensor_msgs
  arc_interfaces
  ament_index_cpp
)
set_property(TARGET codeword_publisher PROPERTY CXX_STANDARD 17)
set_target_properties(codeword_publisher PROPERTIES
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS codeword_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES enc.trt
  DESTINATION share/${PROJECT_NAME}/checkpoints
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
