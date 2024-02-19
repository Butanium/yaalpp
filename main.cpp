#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <argparse/argparse.hpp>

#include "physics/quadtree.hpp"
#include "physics/rect.hpp"
#include <omp.h>
#include <cstdio>
#include "simulation/Environment.h"
#include "Constants.h"
/* Rough pseudo-code:
 * Tensor map = zeros({1000, 1000, 5});
decays = [0, 0, 0, 0.9, 0.5]
void update(State& state) {
    // Update worldObjects (creatures, food, etc.)
    for (object : *objects) {
        object.update(*this, state);
    }
    resolve_physics(objects);

    // Decay
    map *= decays;
    map.diffuse_pheromones(diffusion_rate);

    // Add objects to map
    for (object : *objects) {
        object.add_to_map(*this); // Add the object color and pheromones to the map
    }

    // Clamp max values
    map.clamp_max_tensor_(max_values);
    ffmpeg.write(map);
}


void object.update(world) {
    world.get_view(position, field_of_view);
    actions = this.brain.forward(view);
    this.update_state(actions);  // Change direction, speed, etc.
    this.update_transform();   // Grow, move, etc.
}

void resolve_physics(objects) {
    // Resolve collisions
    for (object : *objects) {
        for (other : *objects) {
            if (object.collides(other)) {
                object.resolve_collision(other);
            }
        }
    }
}

void map.diffuse_pheromones(float diffusion_rate) {
    // Diffuse pheromones
    for (x : 0..1000) {
        for (y : 0..1000) {
            for (c : 0..3) {
                map[x, y, c] = (map[x, y, c] + map[x-1, y, c] + map[x+1, y, c] + map[x, y-1, c] + map[x, y+1, c]) / (5 + diffusion_rate);
            }
        }
    }
}

void brain.forward(view) {
    return matmul(relu(matmul(view, weights1)), weights2);
}
 */
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
                    5).scan<'i', int>(),
            program.add_argument("-D", "--decay-factors").help("Decay factors for each channel").nargs(
                    argparse::nargs_pattern::at_least_one).scan<'f', float>().default_value(
                    std::vector<float>{0, 0, 0, 0.9, 0.5}
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

inline auto getSlice(Eigen::Tensor<float, 2> &a,
                     Eigen::array<Eigen::Index, 2> &offset,
                     Eigen::array<Eigen::Index, 2> &extent) {
    return a.slice(offset, extent);
}

void test(auto slice) {
    slice.setConstant(1);
}

int main(int argc, char *argv[]) {
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
    auto env = Environment(width, height, num_channels, decay_factors, max_values);
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


}