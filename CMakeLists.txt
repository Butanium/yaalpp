cmake_minimum_required(VERSION 3.20)
# Try to enable CUDA, but make it optional
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA support enabled")
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA not found - building CPU-only version")
endif()
project(yaalpp LANGUAGES CXX)
Include(FetchContent)
# for performance:
add_compile_options(-O3 -march=native)
add_compile_options(-fopenmp -std=c++20)
# add_compile_options(-Wall) # for all warnings
# for debug: add_compile_options(-O0)
find_package(Boost COMPONENTS serialization mpi)
# if not found, download it
if(NOT Boost_FOUND)
        set(BOOST_INCLUDE_LIBRARIES serialization mpi)
        set(BOOST_ENABLE_CMAKE ON)
        FetchContent_Declare(
                Boost
                URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.7z
                USES_TERMINAL_DOWNLOAD TRUE
                DOWNLOAD_NO_EXTRACT FALSE
              )
        FetchContent_MakeAvailable(Boost)
        set(Boost_LIBRARIES ${BOOST_INCLUDE_LIBRARIES})
endif()
# include the boost headers
include_directories(${Boost_INCLUDE_DIRS})

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
# Check if OpenCV is installed and if not use /users/eleves-b/2024/clement.dumas/build/OpenCVConfig.cmake
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
  set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build")
  find_package(OpenCV REQUIRED NO_MODULE)
endif()
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )
# OpenMPI
find_package(MPI REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

set(COMMON_SOURCES
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
        video/stream.cpp
        video/stream.h
        utils/utils.cpp
        utils/utils.h
        entity/plant.cpp
        entity/plant.hpp
        utils/save.hpp
        utils/save.cpp
        utils/statistics.hpp
)

if(HAVE_CUDA)
    list(APPEND COMMON_SOURCES
        topology/topology.cu
        topology/topology.h
        cuda/diffusion_gpu.cu
        cuda/diffusion_gpu.h
    )
endif()

add_executable(tests tests/test.cpp ${COMMON_SOURCES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen ${OpenCV_LIBS} MPI::MPI_CXX OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json Boost::serialization) 
# Boost::mpi)
#${Boost_LIBRARIES})

set(CMAKE_CXX_STANDARD 20)

add_executable(yaalpp main.cpp ${COMMON_SOURCES})
target_link_libraries(yaalpp PRIVATE Eigen3::Eigen argparse ${OpenCV_LIBS} MPI::MPI_CXX OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json Boost::serialization)
# ${Boost_LIBRARIES})