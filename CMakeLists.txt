cmake_minimum_required(VERSION 3.27)
project(yaalpp)
Include(FetchContent)
# for performance: add_compile_options(-O3 -march=native)
# for debug: add_compile_options(-O0)

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
# Avoid parallelization for Eigen
add_compile_definitions(EIGEN_DONT_PARALLELIZE)

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

# OpenMP
find_package(OpenMP REQUIRED)

add_executable(tests tests/test.cpp
        physics/circle.cpp
        physics/circle.cpp
        physics/circle.hpp
        physics/quadtree.cpp
        physics/quadtree.hpp
        physics/rect.cpp
        physics/rect.hpp
        entity/Yaal.h
        entity/Yaal.cpp
        Constants.h
        video/stream.cpp
        video/stream.h
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen ${OpenCV_LIBS} MPI::MPI_CXX OpenMP::OpenMP_CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(yaalpp main.cpp
        physics/circle.cpp
        physics/circle.hpp
        physics/quadtree.cpp
        physics/quadtree.hpp
        physics/rect.cpp
        physics/rect.hpp
        entity/Yaal.cpp
        entity/Yaal.h
        Constants.h
        video/stream.cpp
        video/stream.h
)
target_link_libraries(yaalpp PRIVATE Eigen3::Eigen argparse ${OpenCV_LIBS} MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(example examples/tensor_broadcast_mul.cpp
        physics/circle.cpp
        physics/circle.hpp
        physics/rect.cpp
        physics/rect.hpp
)
target_link_libraries(example PRIVATE Eigen3::Eigen)
