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
#include <filesystem>
#include <sstream>
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
                    std::vector<float>{0, 0, 0, 0.99, 0.98, 0.97}
            ),
            program.add_argument("-d", "--diffusion-rate").help("Diffusion rate for channel").nargs(
                            argparse::nargs_pattern::at_least_one)
                    .scan<'f', float>().default_value(std::vector<float>{0, 0, 0, 10, 19}),
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
            program.add_argument("--num-plants").help(
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
    bool record_snapshot = !program.get<bool>("--no-snapshot");
    int snapshot_interval = program.get<int>("--snapshot-interval");
    std::string name = program.get<std::string>("--name");
    bool cuda = program.get<bool>("--cuda");
    bool cpu = program.get<bool>("--cpu");
    if (cuda && cpu) {
        std::cerr << "Cannot use both CUDA and CPU" << std::endl;
        MPI_Finalize();
        return 1;
    }

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
    Vec2 top_left_position((float) column * sub_width, (float) row * sub_height);
    if (row == rows - 1) {
        sub_height += height % sub_height;
    }
    if (column == columns - 1) {
        sub_width += width % sub_width;
    }
    int sub_num_yaal = num_yaals / num_chunks;
    int sub_num_plant = num_plants / num_chunks;
    if (mpi_rank == 0) {
        sub_num_yaal += num_yaals % num_chunks;
        sub_num_plant += num_plants % num_chunks;
    }
    auto env = Environment(sub_height, sub_width, num_channels, decay_factors, diffusion_rate, max_values,
                           std::move(top_left_position), rows, columns);
    if (cuda) {
      env.diffusion_filter.use_cuda = true;
      std::cout << "Using CUDA for diffusion" << std::endl;
    } else if (cpu) {
      env.diffusion_filter.use_cuda = false;
      std::cout << "Using CPU for diffusion" << std::endl;
    } else if (top.gpus > 0) {
      env.diffusion_filter.use_cuda = true;
      std::cout << "Using CUDA for diffusion" << std::endl;
    } else {
      env.diffusion_filter.use_cuda = false;
      std::cout << "Using CPU for diffusion" << std::endl;
    }

    path save = name;
    stringstream ss;
    ss << "env_" << mpi_rank;
    path save_env = save / ss.str();
    path save_env_frames = save_env / "frames";
    path save_global_frames = save / "frames";
    if (mpi_rank == 0) {
        remove_directory_recursively(save);
        ensure_directory_exists(save_global_frames);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ensure_directory_exists(save_env_frames);
    env.create_yaals_and_plants(sub_num_yaal, sub_num_plant);
    // Initialize the streams 
    Stream global_stream((save / "global.mp4").c_str(),  5, cv::Size(width, height), rows, columns, false,
                             MPI_COMM_WORLD);
    MPI_Comm solo_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_rank, 0, &solo_comm);
    Stream local_stream((save_env / "local.mp4").c_str(), 5, cv::Size(sub_width, sub_height), 1, 1, false, solo_comm);
    // start time
    double start = MPI_Wtime();
    for (int i = 0; i < timesteps; i++) {
        if (i % snapshot_interval == 0 && record_snapshot) {
            stringstream ss_frame;
            ss_frame << "frame_" << i << ".png";
            Tensor<float, 3> map = env.real_map(1);
            global_stream.append_frame(map, (save_global_frames / ss_frame.str()).c_str());
            local_stream.append_frame(map, (save_env_frames / ss_frame.str()).c_str());
        }
        env.step();
    }
    double end_time = MPI_Wtime();
    double elapsed = end_time - start;
    if (mpi_rank == 0) {
        std::cout << "Elapsed time: " << elapsed << " seconds" << std::endl;
        std::cout << "Timesteps per second: " << (double) timesteps / elapsed << std::endl;
    }
    
    MPI_Comm_free(&solo_comm);
    MPI_Finalize();
}
