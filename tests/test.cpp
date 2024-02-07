#include <catch2/catch_test_macros.hpp>
#include "../Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <iostream>
#include "../Constants.h"
#include "../physics/quadtree.hpp"
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
    }
    SECTION("Check direction") {
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
        REQUIRE(direction.isApprox(Vec2(0,1)));
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
        REQUIRE(direction.isApprox(Vec2(-1,1).normalized()));
        // Increase the value of the left attraction
        input_view(2, 0, 0) = 2;
        direction = mlp.evaluate(input_view).direction;
        REQUIRE(direction.isApprox(Vec2(-2,1).normalized()));
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

TEST_CASE("Checking validity of quadtree closest neighbor search :") {
    int n_points = 1000;
    std::mt19937 generator;
    auto x_distr = std::uniform_real_distribution<float>(0, (float) 1);
    auto y_distr = std::uniform_real_distribution<float>(0, (float) 1);

    Rect rect(Vec2(0, 0), Vec2(1, 1));
    QuadTree quadTree(rect, 4);

    std::vector<Vec2> points;
    points.reserve(n_points);
    for (int i = 0; i < n_points; i++) {
        points.emplace_back(x_distr(generator), y_distr(generator));
    }

    for (const Vec2& v: points) {
        quadTree.insert(v);
    }

    std::vector<Vec2> closests;
    closests.reserve(n_points);
    for (const Vec2 &v: points) {
        std::optional<Vec2> closest = quadTree.closest(v);
        REQUIRE(closest.has_value());
        if (closest.has_value()) {
            closests.push_back(closest.value());
        } else {
            closests.emplace_back(-1, -1);
        }
    }

    std::vector<Vec2> trueClosests;
    trueClosests.reserve(n_points);
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
        REQUIRE(bestDist > 0);
        if (bestDist > 0)
        {
            trueClosests.push_back(bestPoint);
        }
        else
        {
            trueClosests.emplace_back(-1, -1);
        }
    }

    for (int i = 0; i < n_points; i++) {
        REQUIRE(closests[i] == trueClosests[i]);
    }
}