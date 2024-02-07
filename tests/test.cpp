#include <catch2/catch_test_macros.hpp>
#include "../Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include "../Constants.h"

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