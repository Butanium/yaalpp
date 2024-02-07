cmake_minimum_required(VERSION 3.27)
project(yaalpp)
Include(FetchContent)
#add_compile_options(-O3 -march=native)
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

add_executable(tests tests/test.cpp
        physics/circle.cpp
        physics/circle.hpp
        physics/collision.cpp
        physics/quadtree.cpp
        physics/quadtree.hpp
        physics/rect.cpp
        physics/rect.hpp
        Yaal.h
        Yaal.cpp
        Constants.h
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)

set(CMAKE_CXX_STANDARD 20)

add_executable(yaalpp main.cpp
        physics/circle.cpp
        physics/circle.hpp
        physics/collision.cpp
        physics/quadtree.cpp
        physics/quadtree.hpp
        physics/rect.cpp
        physics/rect.hpp
        Yaal.cpp
        Yaal.h
        Constants.h
)
target_link_libraries(yaalpp PRIVATE Eigen3::Eigen argparse)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yaalpp PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(example examples/tensor_broadcast_mul.cpp
        physics/circle.cpp
        physics/circle.hpp
        physics/rect.cpp
        physics/rect.hpp
)
target_link_libraries(example PRIVATE Eigen3::Eigen)