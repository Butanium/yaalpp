#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <argparse/argparse.hpp>

#include "physics/quadtree.hpp"
#include "physics/rect.hpp"
#include "diffusion/separablefilter.hpp"
#include "diffusion/badbadbad.hpp"
#include <omp.h>
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

using Vec2 = Eigen::Vector2f;

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
#ifdef _OPENMP
    std::cout << "OpenMP is enabled" << std::endl;
#endif
    int seed = 42;
    int n_points = 100000;
    std::mt19937 generator(seed);
    auto x_distr = std::uniform_real_distribution<float>(0, (float) 1);
    auto y_distr = std::uniform_real_distribution<float>(0, (float) 1);

    Rect rect(Vec2(0, 0), Vec2(1, 1));
    QuadTree quadTree(std::move(rect), 4);

    float t1 = omp_get_wtime();
    Vec2* points;
    points = (Vec2*) malloc(n_points * sizeof(Vec2));
    for (int i = 0; i < n_points; i++) {
        points[i] = Vec2(x_distr(generator), y_distr(generator));
    }
    float t2 = omp_get_wtime();
    std::cout << "Creation took: " << t2 - t1 << " seconds" << std::endl;

    float t3 = omp_get_wtime();
#pragma omp parallel for default(none) shared(quadTree, points, n_points) schedule(static) num_threads(8)
    for (int i = 0; i < n_points; i++) {
        quadTree.insert(points[i]);
    }
    float t4 = omp_get_wtime();
    std::cout << "Insertion took: " << t4 - t3 << " seconds" << std::endl;

    float t5 = omp_get_wtime();
    Vec2 *closests = new Vec2[n_points];
    int errors = 0;
#pragma omp parallel for default(none) shared(quadTree, points, closests, n_points, errors) schedule(static) num_threads(8)
    for (int i = 0; i < n_points; i++) {
        auto v = points[i];
        std::optional<Vec2> closest = quadTree.closest(v);
        if (closest.has_value()) {
            closests[i] = closest.value();
        } else {
            // Yes this is not thread safe, but who cares (Catch2 requires are not thread safe)
            errors++;
        }
    }
    float t6 = omp_get_wtime();
    std::cout << "Closest recursive took: " << t6 - t5 << " seconds and had " << errors << " errors" << std::endl;

    float t7 = omp_get_wtime();
    Vec2 *closests2 = new Vec2[n_points];
    errors = 0;
    for (int i = 0; i < n_points; i++) {
        auto v = points[i];
        std::optional<Vec2> closest = quadTree.closestIterative(v);
        if (closest.has_value()) {
            closests2[i] = closest.value();
        } else {
            // Yes this is not thread safe, but who cares (Catch2 requires are not thread safe)
            errors++;
        }
    }
    float t8 = omp_get_wtime();
    std::cout << "Closest iterative took: " << t8 - t7 << " seconds and had " << errors << " errors" << std::endl;

    free(points);
    delete[] closests;
    delete[] closests2;

    /* test diffusion
     * */

    float t9 = omp_get_wtime();

    int h = 1080;
    int w = 1920;
    int c = 7;
    int filter_size = 3;

    SeparableFilter filter(filter_size, c);

    Tensor<float, 3> input(h, w, c);
    input.setRandom();
    float t10 = omp_get_wtime();
    std::cout << "Map creation took: " << t10 - t9 << " seconds" << std::endl;

    float t11 = omp_get_wtime();
    Tensor<float, 3> output(h, w, c);
    filter.apply(input, output);
    float t12 = omp_get_wtime();
    std::cout << "Diffusion took: " << t12 - t11 << " seconds" << std::endl;

    float t15 = omp_get_wtime();
    float diffusion_rates[c];
    for (int i = 0; i < c; i++) {
        diffusion_rates[i] = 1;
    }
    diffuse_pheromones(input, diffusion_rates, output);
    float t16 = omp_get_wtime();
    std::cout << "Diffusion naive took: " << t16 - t15 << " seconds" << std::endl;
    std::cout << "Naive speedup: " << (t16 - t15) / (t12 - t11) << " times" << std::endl;

    float t13 = omp_get_wtime();
    filter.apply_inplace(input);
    float t14 = omp_get_wtime();
    std::cout << "Diffusion inplace took: " << t14 - t13 << " seconds" << std::endl;
    std::cout << "Inplace speedup: " << (t12 - t11) / (t14 - t13) << " times" << std::endl;

    /* correction of diffusion
     * */
    int minih = 4;
    int miniw = 5;
    int minic = 1;
    int mini_filter_size = 5;

    SeparableFilter mini_filter(mini_filter_size, minic);

    Tensor<float, 3> mini_input(minih, miniw, minic);
    mini_input.setZero();
    mini_input(0, 0, 0) = 1;
    mini_input(0, 1, 0) = 1;
    mini_input(1, 0, 0) = 1;

    std::cout << "Input: " << std::endl << mini_input.reshape(array<int, 2>{minih, miniw * minic}) << std::endl;

    Tensor<float, 3> mini_output(minih, miniw, minic);
    filter.apply(mini_input, mini_output);
    std::cout << "Output: " << std::endl << mini_output.reshape(array<int, 2>{minih, miniw * minic}) << std::endl;

    filter.apply_inplace(mini_input);
    std::cout << "Inplace: " << std::endl << mini_input.reshape(array<int, 2>{minih, miniw * minic}) << std::endl;

    return 0;
}