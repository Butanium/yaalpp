#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <argparse/argparse.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

#include <mpi.h>
#include "video/stream.h"

#include "physics/quadtree.hpp"
#include "utils/rect.hpp"
#include <omp.h>
#include <cstdio>
#include "simulation/Environment.h"
#include "Constants.h"

using json = nlohmann::json;

#include "topology/topology.h"

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
            program.add_argument("--allow-idle-process").help("Experimental grid attribution").flag()
    };
    help_groups["Simulation parameters"] = {
//TODO:            program.add_argument("-l", "--load").help(
//                    "Load a simulation from a file. Ignores other simulation parameters"),
            program.add_argument("-H", "--height").help("Height of the map").default_value(1000).scan<'i', int>(),
            program.add_argument("-W", "--width").help("Width of the map").default_value(1000).scan<'i', int>(),
            program.add_argument("-C", "--channels").help("Number of channels in the map").default_value(
                    6).scan<'i', int>(),
            program.add_argument("-n", "--num-yaals").help(
                    "Number of yaals at the start of the simulation").default_value(100).scan<'i', int>(),
            program.add_argument("-n", "--num-plants").help(
                    "Number of plants at the start of the simulation").default_value(200).scan<'i', int>(),
            program.add_argument("-t", "--timesteps").help("Number of timesteps to simulate").default_value(
                    10000).scan<'i', int>(),
            program.add_argument("-s", "--seed").help("The seed of the simulation").scan<'i', int>()
    };
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cout << program.usage() << std::endl;
        print_grouped_help(help_groups);
        std::cerr << err.what() << std::endl;
        exit(1);
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
    int num_plants = program.get<int>("--num-plants");
    int timesteps = program.get<int>("--timesteps");
    bool allow_idle = program.get<bool>("--allow-idle-process");

    Topology top = get_topology(MPI_COMM_WORLD);
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (program.present<int>("--seed")) {
        int seed = program.get<int>("--seed");
        Yaal::generator.seed(seed + mpi_rank);
        YaalGenome::generator.seed(seed + mpi_rank);
    }
    std::cout << "Hello from rank!" << mpi_rank << " of " << top.processes << " processes" << std::endl;
    std::cout << "Running with " << top.cores_per_process << " cores per process and " << top.gpus << " gpus"
              << " with a total of " << top.gpu_memory << "MB of GPU memory" << std::endl;
    std::cout << "Running with a total of " << top.nodes << " nodes" << std::endl;
    // Divide the environment into subenvironments
    auto [rows, columns] = grid_decomposition(top.processes, allow_idle);
    if (rows > columns && width > height) {
        int temp = rows;
        rows = columns;
        columns = temp;
    }
    assert(allow_idle || (columns * rows == top.processes));
    if (columns * rows <= mpi_rank) {
        std::cout << "Rank " << mpi_rank << " is idle" << std::endl;
        MPI_Finalize();
        return 0;
    }
    int num_chunks = rows * columns;

    int sub_height = height / rows;
    int sub_width = width / columns;
    int row = mpi_rank / columns;
    int column = mpi_rank % columns;
    int left_neighbor = mpi_rank - 1;
    int right_neighbor = mpi_rank + 1;
    int top_neighbor = mpi_rank - columns;
    int bottom_neighbor = mpi_rank + columns;
    auto view_offset = Offset::zero();
    auto share_offset = Offset::zero();
    if (column == 0) {
        view_offset.left = Constants::Yaal::MAX_FIELD_OF_VIEW;
    } else {
        share_offset.left = Constants::Environment::SHARED_SIZE;
    }
    if (row == 0) {
        view_offset.top = Constants::Yaal::MAX_FIELD_OF_VIEW;
    } else {
        share_offset.top = Constants::Environment::SHARED_SIZE;
    }
    if (row == rows - 1) {
        sub_height += height % sub_height;
        view_offset.bottom = Constants::Yaal::MAX_FIELD_OF_VIEW;
    } else {
        share_offset.bottom = Constants::Environment::SHARED_SIZE;
    }
    if (column == columns - 1) {
        sub_width += width % sub_width;
        view_offset.right = Constants::Yaal::MAX_FIELD_OF_VIEW;
    } else {
        share_offset.right = Constants::Environment::SHARED_SIZE;
    }
    int sub_num_yaal = num_yaals / num_chunks;
    int sub_num_plant = num_plants / num_chunks;
    if (mpi_rank == 0) {
        sub_num_yaal += num_yaals % num_chunks;
        sub_num_plant += num_plants % num_chunks;
    }
    std::cout << "Subenvironment " << mpi_rank << " at " << row << " " << column << " with size " << sub_height << " "
              << sub_width << std::endl;
    auto env = Environment(sub_height, sub_width, num_channels, decay_factors, diffusion_rate, max_values);
    env.yaals.reserve(num_yaals);
    for (int i = 0; i < num_yaals; i++) {
        Yaal yaal = Yaal::random(num_channels);
        int ms = Constants::Yaal::MAX_SIZE;
        yaal.set_random_position(Vec2(ms, ms), Vec2(width - ms, height - ms));
        env.yaals.push_back(yaal);
    }
    for (int t = 0; t < timesteps; t++) {
        std::cout << "#";
        env.step();
    }
    MPI_Finalize();
}