cmake_minimum_required(VERSION 3.27)
project(yaalpp)
Include(FetchContent)

# Test library
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)

# Tensor library
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
)
FetchContent_MakeAvailable(Eigen)

# Argument parser library
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )

# OpenMPI
find_package(MPI REQUIRED)

add_executable(tests tests/test.cpp
  video/stream.cpp
  video/stream.h
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen ${OpenCV_LIBS} MPI::MPI_CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(yaalpp main.cpp
  video/stream.cpp
  video/stream.h
)
target_link_libraries(yaalpp PRIVATE Eigen3::Eigen argparse ${OpenCV_LIBS} MPI::MPI_CXX)

add_executable(example examples/tensor_broadcast_mul.cpp)
target_link_libraries(example PRIVATE Eigen3::Eigen)
