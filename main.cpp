#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <argparse/argparse.hpp>

#include <mpi.h>
#include "video/stream.h"

#include "physics/quadtree.hpp"
#include "utils/rect.hpp"
#include <omp.h>
#include <cstdio>
#include "simulation/Environment.h"
#include "Constants.h"

using Eigen::Tensor;
using Eigen::array;

using Vec2 = Eigen::Vector2f;

using OptionGroup = std::vector<argparse::Argument>;
using GroupsMap = std::map<std::string, OptionGroup>;
using KeyOGPair = std::pair<std::string, OptionGroup>;

void print_grouped_help(const GroupsMap &groups) {
    for (auto &group: groups) {
        std::cout << group.first << ':' << std::endl;
        for (auto &arg: group.second) {
            std::cout << arg;
        }
        std::cout << std::endl;
    }
}


void parse_arguments(int argc, char *argv[], argparse::ArgumentParser &program) {
    GroupsMap help_groups{
            KeyOGPair{"General", {}},
            KeyOGPair{"Simulation parameters", {}},
            KeyOGPair{"Environment Hyperparameters", {}},
    };
    program.add_description("Yet Another Artificial Life Program in cpp");
    help_groups["Environment Hyperparameters"] = {
            program.add_argument("-H", "--height").help("Height of the map").default_value(1000).scan<'i', int>(),
            program.add_argument("-W", "--width").help("Width of the map").default_value(1000).scan<'i', int>(),
            program.add_argument("-C", "--channels").help("Number of channels in the map").default_value(
                    6).scan<'i', int>(),
            program.add_argument("-D", "--decay-factors").help("Decay factors for each channel").nargs(
                    argparse::nargs_pattern::at_least_one).scan<'f', float>().default_value(
                    std::vector<float>{0, 0, 0, 0.9, 0.8, 0.5}
            ),
            program.add_argument("-d", "--diffusion-rate").help("Diffusion rate for channel").nargs(
                            argparse::nargs_pattern::at_least_one)
                    .scan<'f', float>().default_value(std::vector<float>{0, 0, 0, 0.1, 0.9}),
            program.add_argument("-m", "--max-values").help("Max values for each channel").nargs(
                    argparse::nargs_pattern::at_least_one).scan<'f', float>().default_value(
                    std::vector<float>{1, 1, 1, 5, 5})
    };
    help_groups["General"] = {
            program.add_argument("-h", "--help").help("Print this help").nargs(0),
    };
    help_groups["Simulation parameters"] = {
            program.add_argument("-n", "--num-yaals").help(
                    "Number of yaals at the start of the simulation").default_value(100).scan<'i', int>(),
            program.add_argument("-t", "--timesteps").help("Number of timesteps to simulate").default_value(
                    10000).scan<'i', int>(),
    };
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cout << program.usage() << std::endl;
        print_grouped_help(help_groups);
        std::cerr << err.what() << std::endl;
        std::exit(1);
    }
    if (program.is_used("--help")) {
        std::cout << program.usage() << std::endl;
        print_grouped_help(help_groups);
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    argparse::ArgumentParser program("yaalpp");
    parse_arguments(argc, argv, program);
    int height = program.get<int>("--height");
    int width = program.get<int>("--width");
    int num_channels = program.get<int>("--channels");
    auto decay_factors = program.get<std::vector<float>>("--decay-factors");
    auto diffusion_rate = program.get<std::vector<float>>("--diffusion-rate");
    auto max_values = program.get<std::vector<float>>("--max-values");
    int num_yaals = program.get<int>("--num-yaals");
    int timesteps = program.get<int>("--timesteps");
#ifdef _OPENMP
    std::cout << "OpenMP is enabled" << std::endl;
#else
    std::cout << "OpenMP is disabled" << std::endl;
#endif
    auto env = Environment(height, width, num_channels, decay_factors, max_values);
    env.yaals.reserve(num_yaals);
    for (int i = 0; i < num_yaals; i++) {
        Yaal yaal = Yaal::random(num_channels);
        int ms = Constants::Yaal::MAX_SIZE;
        yaal.setRandomPosition(Vec2(ms, ms), Vec2(width - ms, height - ms));
        env.yaals.push_back(yaal);
    }
    for (int t = 0; t < timesteps; t++) {
        std::cout << "#";
        env.step();
    }
    MPI_Finalize();
}