#include <catch2/catch_test_macros.hpp>
#include "../entity/Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <iostream>
#include "../Constants.h"
#include "../physics/quadtree.hpp"
#include <catch2/catch_get_random_seed.hpp>

using Eigen::Tensor;

void eval_yaal() {
    const Vec2 pos = Vec2::Zero();
    Yaal yaal = Yaal::random(3);
    int view_size = yaal.genome.field_of_view * 2 + yaal.genome.size;
    Tensor<float, 3> input_view(view_size, view_size, 3);
    input_view.setRandom();
    yaal.update(input_view);
}


TEST_CASE("Yaal eval") {
    SECTION("eval random yaals") {
        for (int i = 0; i < 1000; i++)
            eval_yaal();
    }SECTION("Check direction") {
        Tensor<float, 1> direction_weights(3);
        direction_weights.setZero();
        direction_weights(0) = 1;
        Yaal yaal = Yaal::random(3);
        auto mlp = yaal.genome.brain;
        mlp.direction_weights = direction_weights;
        Tensor<float, 3> input_view(5, 5, 3);
        input_view.setZero();
        // Add a fake attraction
        input_view(0, 2, 1) = 1;
        auto direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isMuchSmallerThan(Constants::EPSILON));
        // Add a real attraction to the top
        input_view(0, 2, 0) = 1;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox(Vec2(0, 1)));
        // Add a real attraction to the bottom
        input_view(4, 2, 0) = 1;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isMuchSmallerThan(Constants::EPSILON));
        // Add a real attraction to the left
        input_view(2, 0, 0) = 1;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox(Vec2(-1, 0)));
        // Increase the value of the top attraction
        input_view(0, 2, 0) = 2;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox(Vec2(-1, 1).normalized()));
        // Increase the value of the left attraction
        input_view(2, 0, 0) = 2;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox(Vec2(-2, 1).normalized()));
        input_view.setZero();
        input_view(0, 0, 0) = 1;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox(Vec2(-1, 1).normalized()));
        // Add a repulsion to the top
        input_view(0, 2, 0) = -1;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox((Vec2(-1, 1).normalized() + Vec2(0, -1)).normalized()));
    }
}

void test_quadtree(int n_points, int n_threads, unsigned int seed) {
    std::mt19937 generator(seed);
    auto x_distr = std::uniform_real_distribution<float>(0, (float) 1);
    auto y_distr = std::uniform_real_distribution<float>(0, (float) 1);

    Rect rect(Vec2(0, 0), Vec2(1, 1));
    QuadTree quadTree(std::move(rect), 4);

    std::vector<Vec2> points;
    points.reserve(n_points);
    for (int i = 0; i < n_points; i++) {
        points.emplace_back(x_distr(generator), y_distr(generator));
    }
#pragma omp parallel for default(none) shared(quadTree, points) schedule(static) num_threads(n_threads)
    for (const Vec2 &v: points) {
        quadTree.insert(v);
    }

    Vec2 *closests = new Vec2[n_points];
#pragma omp parallel for default(none) shared(quadTree, points, closests, n_points) schedule(static) num_threads(n_threads)
    for (int i = 0; i < n_points; i++) {
        auto v = points[i];
        std::optional<Vec2> closest = quadTree.closest(v);
        REQUIRE(closest.has_value());
        if (closest.has_value()) {
            closests[i] = closest.value();
        }
    }

    std::vector<Vec2> trueClosests;
    trueClosests.reserve(n_points);
#pragma omp parallel for default(none) shared(points, trueClosests, n_points) schedule(static) num_threads(n_threads)
    for (int i = 0; i < n_points; i++) {
        float dist;
        auto v = points[i];
        float bestDist = -1;
        Vec2 bestPoint;
        for (const Vec2 &v2: points) {
            dist = (v - v2).squaredNorm();
            if ((dist < bestDist || bestDist < 0) && dist > 0) {
                bestDist = dist;
                bestPoint = v2;
            }
        }
        REQUIRE(bestDist > 0);
        if (bestDist > 0) {
            trueClosests[i] = bestPoint;
        }
    }

    for (int i = 0; i < n_points; i++) {
        REQUIRE(closests[i] == trueClosests[i]);
    }
    free(closests);
}

TEST_CASE("Checking validity of quadtree closest neighbor search :") {
// Get seed from catch 2
    auto seed = Catch::getSeed();
    SECTION("1 thread") {
        test_quadtree(5000, 1, seed);
    }SECTION("2 threads") {
        test_quadtree(5000, 2, seed);
    }SECTION("3 threads") {
        test_quadtree(5000, 3, seed);
    }SECTION("16 threads") {
        test_quadtree(5000, 16, seed);
    }
}