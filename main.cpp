#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <argparse/argparse.hpp>
#include <cstdio>
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

void parse_arguments(int argc, char *argv[], argparse::ArgumentParser &program) {
    program.add_description("Yet Another Artificial Life Program in cpp");
    program.add_argument("-H", "--height").help("Height of the map").default_value(1000).scan<'i', int>();
    program.add_argument("-W", "--width").help("Width of the map").default_value(1000).scan<'i', int>();
    program.add_argument("-C", "--channels").help("Number of channels in the map").default_value(5).scan<'i', int>();
    program.add_argument("-D", "--decay-factors").help("Decay factors for each channel").nargs(
            argparse::nargs_pattern::at_least_one).scan<'f', float>().default_value(
            std::vector<float>{0, 0, 0, 0.9, 0.5}
    );
    program.add_argument("-d", "--diffusion-rate").help("Diffusion rate for pheromones").nargs(
                    argparse::nargs_pattern::at_least_one)
            .scan<'f', float>();
    program.add_argument("-m", "--max-values").help("Max values for each channel").nargs(
            argparse::nargs_pattern::at_least_one).scan<'f', float>();
    program.parse_args(argc, argv);
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("yaalpp");
    parse_arguments(argc, argv, program);
    int height = program.get<int>("--height");
    int width = program.get<int>("--width");
    int channels = program.get<int>("--channels");
    auto _decay_factors = program.get<std::vector<float>>("--decay-factors");
    Eigen::TensorMap<Tensor<float, 1>> decay_factors_1d(_decay_factors.data(), (long) _decay_factors.size());
    Tensor<float, 3> map(height, width, channels);
    map.setZero();
    auto decay_factors = decay_factors_1d.reshape(array<int, 3>{1, 1, channels}).broadcast(
            array<int, 3>{height, width, 1}).eval();
    map *= decay_factors;
    // Print the 10*10*5 tensor
    std::cout << map.slice(array<Eigen::Index, 3>{0, 0, 0}, array<Eigen::Index, 3>{10, 10, 5})
              << std::endl;
}