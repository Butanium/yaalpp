#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <argparse/argparse.hpp>

#include "physics/quadtree.hpp"
#include "physics/rect.hpp"
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

    /**
     * QuadTree testing
     *
     * Generate 1000 random points, insert them into a quadtree, and query the closest point to each of them
     */

    int n_points = 1000;
    std::mt19937 generator;
    auto x_distr = std::uniform_real_distribution<float>(0, (float) 1);
    auto y_distr = std::uniform_real_distribution<float>(0, (float) 1);

    float t1 = clock();
    Rect rect(Vec2(0, 0), Vec2(1, 1));
    QuadTree quadTree(rect, 4, 0.01f);

    std::vector<Vec2> points;
    points.reserve(n_points);
    for (int i = 0; i < n_points; i++) {
        points.emplace_back(x_distr(generator), y_distr(generator));
    }
    float t2 = clock();
    std::cout << "Generating " << n_points << " random points took " << (t2 - t1) / CLOCKS_PER_SEC << " seconds" << std::endl;

    float t3 = clock();
    for (Vec2 v: points) {
        quadTree.insert(v);
    }
    float t4 = clock();
    std::cout << "Inserting " << n_points << " points into the quadtree took " << (t4 - t3) / CLOCKS_PER_SEC << " seconds"
              << std::endl;

    std::vector<Vec2> closests1;
    closests1.reserve(n_points);
    float t5 = clock();
    for (const Vec2 &v: points) {
        std::optional<Vec2> closest = quadTree.closest(v);
        if (closest.has_value()) {
            closests1.push_back(closest.value());
        } else {
            std::cout << "No closest point found" << std::endl;
        }
    }
    float t6 = clock();
    std::cout << "Closest point search took " << (t6 - t5) / CLOCKS_PER_SEC << " seconds" << std::endl;

    std::vector<Vec2> closests2;
    closests2.reserve(n_points);
    float t7 = clock();
    for (const Vec2 &v: points) {
        std::optional<Vec2> closest = quadTree.naiveClosest(v);
        if (closest.has_value()) {
            closests2.push_back(closest.value());
        } else {
            std::cout << "No closest point found" << std::endl;
        }
    }
    float t8 = clock();
    std::cout << "Naive closest point search took " << (t8 - t7) / CLOCKS_PER_SEC << " seconds" << std::endl;

    // verify that the two closest point searches are the same
    for (int i = 0; i < n_points; i++) {
        if (closests1[i] != closests2[i]) {
            std::cout << "Closest point search results differ by " << (closests1[i] - closests2[i]).norm() << std::endl;
            break;
        }
    }

    std::vector<Vec2> closests3;
    closests3.reserve(n_points);
    float t9 = clock();
    float dist;
    for (const Vec2 &v: points) {
        float bestDist = -1;
        Vec2 bestPoint;
        for (const Vec2 &v2: points) {
            dist = (v - v2).squaredNorm();
            if ((dist < bestDist || bestDist < 0) && dist > 0) {
                bestDist = dist;
                bestPoint = v2;
            }
        }
        closests3.push_back(bestPoint);
    }
    float t10 = clock();
    std::cout << "baseline took " << (t10 - t9) / CLOCKS_PER_SEC << " seconds" << std::endl;

    for (int i = 0; i < n_points; i++) {
        if (closests1[i] != closests3[i]) {
            std::cout << "Closest point search results differ by " << (closests1[i] - closests3[i]).norm() << std::endl;
            break;
        }
    }
}