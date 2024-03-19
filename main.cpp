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
#include <filesystem>
#include <sstream>
#include "Constants.h"
#include "topology/topology.h"
#include "utils/utils.h"

using json = nlohmann::json;
using std::filesystem::path;
using std::stringstream;

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
            program.add_argument("--allow-idle-process").help("Experimental grid attribution").flag(),
            program.add_argument("--snapshot-interval").help("Interval between snapshots").default_value(100).scan<'i', int>(),
            program.add_argument("--no-snapshot").help("Turn off snapshot").flag(),
            program.add_argument("--name").help("Name of the simulation").default_value("simulation"),
            program.add_argument("--cuda").help("Use CUDA for computation").flag(),
            program.add_argument("--cpu").help("Use only CPU for computation").flag()
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
    int snapshot_interval = program.get<int>("--snapshot-interval");
    bool record_snapshot = !program.get<bool>("--no-snapshot");
    std::string name = program.get<std::string>("--name");
    bool cuda = program.get<bool>("--cuda");
    bool cpu = program.get<bool>("--cpu");

    Topology top = get_topology(MPI_COMM_WORLD);

#ifdef _OPENMP
    std::cout << "OpenMP is enabled" << std::endl;
#else
    std::cout << "OpenMP is disabled" << std::endl;
#endif
    auto env = Environment(height, width, num_channels, decay_factors, diffusion_rate, max_values);

    if (cuda) {
      env.diffusion_filter.use_cuda = true;
    } else if (cpu) {
      env.diffusion_filter.use_cuda = false;
    } else if (top.gpus > 0) {
      env.diffusion_filter.use_cuda = true;
    } else {
      env.diffusion_filter.use_cuda = false;
    }

    path save = name;
    stringstream ss;
    path save_global_frames = save / "frames";
    remove_directory_recursively(save);
    ensure_directory_exists(save_global_frames);

    env.create_yaals_and_plants(num_yaals, num_plants);
    // Initialize the streams 
    Stream global_stream((save / "simulation.mp4").c_str(),  5, cv::Size(width, height), 1, 1, false,
                             MPI_COMM_WORLD);
    for (int i = 0; i < timesteps; i++) {
        if (i % snapshot_interval == 0 && record_snapshot) {
            stringstream ss_frame;
            ss_frame << "frame_" << i << ".png";
            Tensor<float, 3> map = env.real_map(1);
            global_stream.append_frame(map, (save_global_frames / ss_frame.str()).c_str());
        }
        env.step();
    }

    MPI_Finalize();
}
