#include <catch2/catch_test_macros.hpp>
#include <random>
#include <iostream>
#include "../physics/quadtree.hpp"

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
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