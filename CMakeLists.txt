cmake_minimum_required(VERSION 3.27)
project(yaalpp)
Include(FetchContent)
# for performance:
#add_compile_options(-O3 -march=native)
add_compile_options(-fopenmp -std=c++20)
add_compile_options(-Wall) # for all warnings
# for debug: add_compile_options(-O0)

# Test library
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# JSON library
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

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
        utils/circle.cpp
        utils/circle.cpp
        utils/circle.hpp
        physics/quadtree.cpp
        physics/quadtree.hpp
        utils/rect.cpp
        utils/rect.hpp
        entity/Yaal.h
        entity/Yaal.cpp
        Constants.h
        simulation/Environment.cpp
        simulation/Environment.h
        diffusion/separablefilter.hpp
        diffusion/separablefilter.cpp
        utils/offset.hpp
        video/stream.cpp
        video/stream.h
        utils/utils.cpp
        utils/utils.h
        entity/plant.cpp
        entity/plant.hpp
        utils/save.hpp
        utils/save.cpp

)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen ${OpenCV_LIBS} MPI::MPI_CXX OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)

set(CMAKE_CXX_STANDARD 20)

add_executable(yaalpp main.cpp
        utils/circle.cpp
        utils/circle.hpp
        physics/quadtree.cpp
        physics/quadtree.hpp
        utils/rect.cpp
        utils/rect.hpp
        entity/Yaal.cpp
        entity/Yaal.h
        Constants.h
        diffusion/separablefilter.hpp
        diffusion/separablefilter.cpp
        simulation/Environment.cpp
        simulation/Environment.h
        utils/offset.hpp
        video/stream.cpp
        video/stream.h
        utils/utils.cpp
        utils/utils.h
        entity/plant.cpp
        entity/plant.hpp
        utils/save.hpp
        utils/save.cpp
)
target_link_libraries(yaalpp PRIVATE Eigen3::Eigen argparse ${OpenCV_LIBS} MPI::MPI_CXX OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)