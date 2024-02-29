#define CATCH_CONFIG_RUNNER

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_console.hpp>
#include <catch2/reporters/catch_reporter_helpers.hpp>

#include <mpi.h>
#include "../video/stream.h"

#include "../entity/Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <iostream>
#include "../Constants.h"
#include "../physics/quadtree.hpp"
#include "../simulation/Environment.h"
#include <catch2/catch_get_random_seed.hpp>

using Eigen::Tensor;
using Eigen::array;
using Eigen::Index;

template<int N>
bool is_close(const Tensor<float, N> &a, const Tensor<float, N> &b) {
    Tensor<bool, 0> res = ((a - b).abs() < Constants::EPSILON).all();
    return res(0);
}


TEST_CASE("Yaal eval") {
    SECTION("eval random yaals") {
        for (int i = 0; i < 100; i++) {
            Yaal yaal = Yaal::random(3);
            int view_size = yaal.genome.field_of_view * 2 + yaal.genome.size;
            Tensor<float, 3> input_view(view_size, view_size, 3);
            input_view.setRandom();
            yaal.update(input_view);
        }
    }SECTION("Direction Matrix") {
        Tensor<float, 3> t(2, 2, 2);
        // Assign values using a comma initializer
        float sqr = 1 / std::sqrt(2);
        t.setValues({{{-sqr, -sqr}, {sqr, -sqr}},
                     {{-sqr, sqr},  {sqr, sqr}}});
        auto dir_matrix = direction_matrix(2, 2);
        REQUIRE(is_close(t, dir_matrix));
        dir_matrix = direction_matrix(3, 3);
        t = Tensor<float, 3>(3, 3, 2);
        t.setValues({{{-sqr, -sqr}, {0, -1}, {sqr, -sqr}},
                     {{-1,   0},    {0, 0},  {1,   0}},
                     {{-sqr, sqr},  {0, 1},  {sqr, sqr}}});
        REQUIRE(is_close(t, dir_matrix));
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
        auto direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isMuchSmallerThan(Constants::EPSILON));
        // Add a real attraction to the top
        input_view(0, 2, 0) = 1;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isApprox(Vec2(0, -1)));
        // Add a real attraction to the bottom
        input_view(4, 2, 0) = 1;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isMuchSmallerThan(Constants::EPSILON));
        // Add a real attraction to the left
        input_view(2, 0, 0) = 1;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isApprox(Vec2(-1, 0)));
        // Increase the value of the top attraction
        input_view(0, 2, 0) = 2;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isApprox(Vec2(-1, -1).normalized()));
        // Increase the value of the left attraction
        input_view(2, 0, 0) = 2;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isApprox(Vec2(-2, -1).normalized()));
        input_view.setZero();
        input_view(0, 0, 0) = 1;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isApprox(Vec2(-1, -1).normalized()));
        // Add a repulsion to the top
        input_view(0, 2, 0) = -1;
        direction = mlp.evaluate(input_view, 5, 5).direction;
        REQUIRE(direction.isApprox((Vec2(-1, -1).normalized() + Vec2(0, 1)).normalized()));
    }
}

// TODO: replace thread unsafe things in this test by or reduction
void test_quadtree(int n_points, int n_threads, unsigned int seed) {
    std::mt19937 generator(seed);
    auto x_distr = std::uniform_real_distribution<float>(0, (float) 1);
    auto y_distr = std::uniform_real_distribution<float>(0, (float) 1);

    Rect rect(Vec2(0, 0), Vec2(1, 1));
    for (int max_capacity = 1; max_capacity < 8; max_capacity += 3) {
        QuadTree quadTree(std::move(rect), max_capacity);

        Vec2 *points;
        points = (Vec2 *) malloc(n_points * sizeof(Vec2));
        for (int i = 0; i < n_points; i++) {
            points[i] = Vec2(x_distr(generator), y_distr(generator));
        }
#pragma omp parallel for default(none) shared(quadTree, points, n_points) schedule(static) num_threads(n_threads)
        for (int i = 0; i < n_points; i++) {
            quadTree.insert(points[i]);
        }

        Vec2 *closests = new Vec2[n_points];
        int errors = 0;
#pragma omp parallel for default(none) shared(quadTree, points, closests, n_points, n_threads, errors) schedule(static) num_threads(n_threads)
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
        REQUIRE(errors == 0);

        int trueErrors = 0;
        Vec2 *trueClosests;
        trueClosests = (Vec2 *) malloc(n_points * sizeof(Vec2));
#pragma omp parallel for default(none) shared(points, trueClosests, n_points, n_threads, trueErrors) schedule(static) num_threads(n_threads)
        for (int i = 0; i < n_points; i++) {
            float dist;
            auto v = points[i];
            float bestDist = -1;
            Vec2 bestPoint;
            for (int j = 0; j < n_points; j++) {
                auto v2 = points[j];
                dist = (v - v2).squaredNorm();
                if ((dist < bestDist || bestDist < 0) && dist > 0) {
                    bestDist = dist;
                    bestPoint = v2;
                }
            }
            if (bestDist > 0) {
                trueClosests[i] = bestPoint;
            } else {
                trueErrors++;
            }
        }
        REQUIRE(trueErrors == 0);

        for (int i = 0; i < n_points; i++) {
            REQUIRE(closests[i] == trueClosests[i]);
        }
        free(closests);
        free(trueClosests);
        free(points);
    }
}


TEST_CASE("Output video 5 processes", "[output_five]") {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 5) {
        return;
    }

    Stream stream("output_5_mpi.mp4", 2, cv::Size(1000, 1000), 2, 2, true, MPI_COMM_WORLD);

    int rank_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    Eigen::Tensor<float, 3> map(2, 2, 3);
    if (rank_id == 0) {
        for (int i = 0; i < 16; i++) stream.append_frame(nullptr);
    } else {
        for (int i = 0; i < 16; i++) {
            map.setZero();

            if ((i / 4) + 1 == rank_id) {
                int x = i % 4;
                map(x % 2, x / 2, 0) = 1;
                map(x % 2, x / 2, 1) = 1;
                map(x % 2, x / 2, 2) = 1;
            }

            stream.append_frame(map);
        }
    }

    stream.end_stream();
}

TEST_CASE("Output video 4 processes", "[output_four]") {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 4) {
        return;
    }

    Stream stream("output_4_mpi.mp4", 2, cv::Size(1000, 1000), 2, 2, false, MPI_COMM_WORLD);

    int rank_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    Eigen::Tensor<float, 3> map(2, 2, 3);

    for (int i = 0; i < 16; i++) {
        map.setZero();

        if (i / 4 == rank_id) {
            int x = i % 4;
            map(x % 2, x / 2, 0) = 1;
            map(x % 2, x / 2, 1) = 1;
            map(x % 2, x / 2, 2) = 1;
        }

        stream.append_frame(map);
    }

    stream.end_stream();
}

TEST_CASE("Output video one process", "[output_single]") {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1) {
        return;
    }

    Stream stream("output_single_mpi.mp4", 2, cv::Size(1000, 1000), 1, 1, false, MPI_COMM_WORLD);
    Eigen::Tensor<float, 3> map(3, 3, 3);

    for (int i = 0; i < 10; i++) {
        map.setZero();

        map(i / 3, i % 3, 0) = 1;
        map(i / 3, i % 3, 1) = 1;
        map(i / 3, i % 3, 2) = 1;

        stream.append_frame(map);
    }
    stream.end_stream();
}

TEST_CASE("Checking validity of quadtree closest neighbor search :") {
// Get seed from catch 2
    auto seed = Catch::getSeed();
    SECTION("7 threads") {
        test_quadtree(5000, 7, seed);
    }SECTION("16 threads") {
        test_quadtree(5000, 16, seed);
    }
}

// Custom main function to handle MPI initialization and finalization
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}


TEST_CASE("ENVIRONMENT") {
    SECTION("Add to map") {
        using Constants::Yaal::MAX_FIELD_OF_VIEW;
        auto decays = std::vector<float>{0.9, 0.9, 0.9, 0.9};
        Environment env(Constants::Yaal::MAX_SIZE, Constants::Yaal::MAX_SIZE, 4, decays, decays, decays);
        Yaal yaal = Yaal::random(4);
        yaal.position = Vec2(1, 1) * Constants::Yaal::MAX_SIZE / 2;  // Center the Yaal
        env.add_to_map(yaal);
        Tensor<float, 3> map(2 * Constants::Yaal::MAX_FIELD_OF_VIEW + Constants::Yaal::MAX_SIZE,
                             2 * Constants::Yaal::MAX_FIELD_OF_VIEW + Constants::Yaal::MAX_SIZE, 4);
        map.setZero();
        map.slice(array<Index, 3>{Constants::Yaal::MAX_FIELD_OF_VIEW, Constants::Yaal::MAX_FIELD_OF_VIEW, 0},
                  array<Index, 3>{Constants::Yaal::MAX_SIZE, Constants::Yaal::MAX_SIZE, 4}) += yaal.genome.body;
        REQUIRE(is_close(env.map, map));
        yaal.genome.field_of_view = MAX_FIELD_OF_VIEW;
        Tensor<float, 3> view = env.get_view(yaal);
        REQUIRE(is_close(env.map, view));
    }SECTION("Env steps") {
        // TODO: this should pass once physics are implemented
        using Constants::Yaal::MAX_SIZE;
        auto seed = Catch::getSeed();
        seed = 3338408716;
        Yaal::generator.seed(seed);
        YaalGenome::generator.seed(seed);
        auto decays = std::vector<float>{0.9, 0.9, 0.9, 0.9};
        std::cerr << "Creating env" << std::endl;
        int height = 30;
        int width = 30;
        Environment env(width, height, 4, decays, decays, decays);
        std::cerr << "Adding yaals" << std::endl;
        for (int i = 0; i < 100; i++) {
            Yaal yaal = Yaal::random(4);
            yaal.setRandomPosition(Vec2(MAX_SIZE, MAX_SIZE), Vec2(width - MAX_SIZE, height - MAX_SIZE));
            env.yaals.push_back(yaal);
            std::cout << yaal.position.x() << " " << yaal.position.y() << std::endl;
            std::cout << yaal.genome.brain.direction_weights << std::endl;
        }
        std::cerr << "Updating yaals" << std::endl;
        for (int i = 0; i < 10; i++) {
            if (i % 10 == 0) {
                std::cerr << i << std::endl;
            }
            env.step();
        }
        // at the end print all positions sorted from top left to bottom right
        std::vector<Vec2> positions;
        for (auto &yaal: env.yaals) {
            positions.push_back(yaal.position);
        }
        std::sort(positions.begin(), positions.end(), [](const Vec2 &a, const Vec2 &b) {
            return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
        });
        for (auto &pos: positions) {
            std::cerr << pos.x() << " " << pos.y() << std::endl;
        }
    }
}