cmake_minimum_required(VERSION 3.18)
project(yolo11_pose_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 90)
endif()

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" CACHE PATH "CUDA Toolkit path")
set(TensorRT_ROOT_DIR "C:/TensorRT/TensorRT-10.9.0.34" CACHE PATH "TensorRT path")

find_package(CUDA REQUIRED)

include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TensorRT_ROOT_DIR}/include
    ${CMAKE_SOURCE_DIR}/common/include
)

link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    ${TensorRT_ROOT_DIR}/lib
)

set(COMMON_SOURCES
    common/src/types.cpp
    common/src/trt_engine.cpp
)

set(COMMON_CUDA_SOURCES
    common/cuda/preprocess.cu
    common/cuda/postprocess.cu
)

add_library(common STATIC ${COMMON_SOURCES} ${COMMON_CUDA_SOURCES})
target_link_libraries(common
    nvinfer_10
    nvonnxparser_10
    cudart
)

set(STREAM_ASYNC_SOURCES
    stream-async/src/main.cpp
    stream-async/src/yolo_pose_stream.cpp
)

add_executable(yolo_stream_async ${STREAM_ASYNC_SOURCES})
target_link_libraries(yolo_stream_async
    common
    nvinfer_10
    nvonnxparser_10
    cudart
)

set(CUDA_GRAPH_SOURCES
    cuda-graph/src/main.cpp
    cuda-graph/src/yolo_pose_graph.cpp
)

add_executable(yolo_cuda_graph ${CUDA_GRAPH_SOURCES})
target_link_libraries(yolo_cuda_graph
    common
    nvinfer_10
    nvonnxparser_10
    cudart
)

set_target_properties(yolo_stream_async yolo_cuda_graph PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "TensorRT: ${TensorRT_ROOT_DIR}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
