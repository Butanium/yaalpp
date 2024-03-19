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
#include "../physics/quadtree.hpp"
#include "../simulation/Environment.h"
#include "../utils/save.hpp"
#include "../utils/utils.h"
#include <catch2/catch_get_random_seed.hpp>
#include <vector>
#include <format>
#include <boost/mpi.hpp>

#define SKIP_IF_NOT_SINGLE_MPI_PROCESS \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    if (rank != 0) { \
        return; \
    }

using Eigen::Tensor;
using Eigen::array;
using Eigen::Index;
using std::filesystem::path;

bool vec_is_close(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < (int) a.size(); i++) {
        if (std::abs(a[i] - b[i]) > Constants::EPSILON) return false;
    }
    return true;
}

TEST_CASE("Topology") {
    Topology t = get_topology(MPI_COMM_WORLD);

    REQUIRE(t.nodes == 1);
    REQUIRE(t.processes == 1);
    REQUIRE(t.cores_per_process == 1);
    REQUIRE(t.gpus == 1);
}

TEST_CASE("Yaal") {
    SKIP_IF_NOT_SINGLE_MPI_PROCESS;
    SECTION("Genome Seeding") {
        auto seed = Catch::getSeed();
        YaalGenome::generator.seed(seed);
        Yaal yaal = Yaal::random(3);
        Yaal yaal2 = Yaal::random(3);
        REQUIRE(!is_close(yaal.genome.brain.direction_weights, yaal2.genome.brain.direction_weights));
        REQUIRE(!vec_is_close(yaal.genome.signature, yaal2.genome.signature));
        YaalGenome::generator.seed(seed);
        Yaal yaal3 = Yaal::random(3);
        REQUIRE(is_close(yaal.genome.brain.direction_weights, yaal3.genome.brain.direction_weights));
        REQUIRE(vec_is_close(yaal.genome.signature, yaal3.genome.signature));
    }SECTION("eval random yaals") {
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

        Vec2 *closests = (Vec2 *) malloc(n_points * sizeof(Vec2));
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


TEST_CASE("Checking validity of quadtree closest neighbor search :") {
    SKIP_IF_NOT_SINGLE_MPI_PROCESS;
    auto seed = Catch::getSeed();
    SECTION("7 threads") {
        test_quadtree(5000, 7, seed);
    }SECTION("16 threads") {
        test_quadtree(5000, 16, seed);
    }
}

TEST_CASE("Output video 5 processes", "[output_five]") {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 5) {
        SKIP("Skipping test, not 5 processes");
    }
    ensure_directory_exists("test_output");

    Stream stream("test_output/output_5_mpi.mp4", 2, cv::Size(1000, 1000), 2, 2, true, MPI_COMM_WORLD);

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
        SKIP("Skipping test, not 4 processes");
    }
    ensure_directory_exists("test_output");
    Stream stream("test_output/output_4_mpi.mp4", 2, cv::Size(1000, 1000), 2, 2, false, MPI_COMM_WORLD);

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
        SKIP("Skipping test, not 1 process");
    }
    ensure_directory_exists("test_output");
    Stream stream("test_output/output_single_mpi.mp4", 2, cv::Size(1000, 1000), 1, 1, false, MPI_COMM_WORLD);
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


// Custom main function to handle MPI initialization and finalization
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}


TEST_CASE("ENVIRONMENT") {
    SKIP_IF_NOT_SINGLE_MPI_PROCESS;
    SECTION("Add to map") {
        using Constants::Yaal::MAX_FIELD_OF_VIEW;

        auto decays = std::vector<float>{0, 0, 0, 0.9};
        auto max_values = std::vector<float>{1, 1, 1, 1};
        Environment env(Constants::Yaal::MAX_SIZE, Constants::Yaal::MAX_SIZE, 4, decays, decays, max_values);
        Yaal yaal = Yaal::random(4);
        yaal.position = Vec2(1, 1) * Constants::Yaal::MAX_SIZE / 2;  // Center the Yaal
        env.add_to_map(yaal);
        Tensor<float, 3> map(2 * Constants::Yaal::MAX_FIELD_OF_VIEW + Constants::Yaal::MAX_SIZE,
                             2 * Constants::Yaal::MAX_FIELD_OF_VIEW + Constants::Yaal::MAX_SIZE, 4);
        map.setZero();
        map.slice(array<Index, 3>{Constants::Yaal::MAX_FIELD_OF_VIEW, Constants::Yaal::MAX_FIELD_OF_VIEW, 0},
                  array<Index, 3>{Constants::Yaal::MAX_SIZE, Constants::Yaal::MAX_SIZE, 4}) += yaal.body;
        REQUIRE(is_close(env.map, map));
        yaal.genome.field_of_view = MAX_FIELD_OF_VIEW;
        Tensor<float, 3> view = env.get_view(yaal);
        REQUIRE(is_close(env.map, view));
    }SECTION("Empty env") {
        std::vector<float> diffusion_factors = {0., 0., 0., 2};
        std::vector<float> max_values = {1, 1, 1, 1};
        std::vector<float> decay_factors = {0, 0, 0, 0.98};
        int height = 500;
        int width = 500;
        Environment env(height, width, 4, decay_factors, diffusion_factors, max_values);
        for (int i = 0; i < 10; i++) {
            env.step();
        }
    }SECTION("No yaal env") {
        std::vector<float> diffusion_factors = {0., 0., 0., 2};
        std::vector<float> max_values = {1, 1, 1, 1};
        std::vector<float> decay_factors = {0, 0, 0, 0.98};
        int height = 500;
        int width = 500;
        int num_plants = 500;
        Environment env(height, width, 4, decay_factors, diffusion_factors, max_values);
        env.create_yaals_and_plants(0, num_plants);
        for (int i = 0; i < 10; i++) {
            env.step();
        }
    }SECTION("Small env") {
        std::vector<float> diffusion_factors = {0., 0., 0., 2};
        std::vector<float> max_values = {1, 1, 1, 1};
        std::vector<float> decay_factors = {0, 0, 0, 0.98};
        int height = 50;
        int width = 50;
        int num_yaal = 5;
        int num_plant = 5;
        Environment env(height, width, 4, decay_factors, diffusion_factors, max_values);
        env.create_yaals_and_plants(num_yaal, num_plant);
        for (int i = 0; i < 10; i++) {
            env.step();
        }
    }SECTION("Env steps and save/load") {
        ensure_directory_exists("test_output/frames");
        using Constants::Yaal::MAX_SIZE;
        auto seed = Catch::getSeed();
        Yaal::generator.seed(seed);
        YaalGenome::generator.seed(seed);
        int num_channels = 4;
        std::vector<float> diffusion_factors = {0., 0., 0., 2};
        std::vector<float> max_values = {1, 1, 1, 1};
        std::vector<float> decay_factors = {0, 0, 0, 0.98};
        int height = 500;
        int width = 500;
        Stream stream("test_output/env_steps.mp4", 10, cv::Size(1000, 1000), 1, 1, false, MPI_COMM_WORLD);
        int num_yaal = 200;
        int num_plant = 200;
        int num_steps = 60;
        Environment env(height, width, num_channels, decay_factors, diffusion_factors, max_values);
        env.create_yaals_and_plants(num_yaal, num_plant);
        std::cout << "Updating yaals" << std::endl;
        remove_files_in_directory("test_output/frames");
        for (int i = 0; i < num_steps; i++) {
            path frame_name = std::format("test_output/frames/frame_{}.png", i);
            // View the env to send the 1, 2, 3 channels instead of 0-3
            auto fov = Constants::Yaal::MAX_FIELD_OF_VIEW;
            Tensor<float, 3> reshaped_map = env.map.slice(array<Index, 3>{fov, fov, 1},
                                                          array<Index, 3>{height, width, 3});
            stream.append_frame(reshaped_map, frame_name.c_str());
            env.step();
        }
        stream.end_stream();
        // Save and load the environment
        std::string save_path = "./save/";
        ensure_directory_exists(save_path);
        save_environment(env, save_path, true);

        Environment env2 = load_environment(save_path);

        for (int i = 0; i < 100; i++) {
            env2.step();
        }
        // save again
        ensure_directory_exists(save_path);
        save_environment(env2, save_path, true);
    } SECTION("Determinism") {
        using Constants::Yaal::MAX_SIZE;
        std::vector<float> diffusion_factors = {0., 0., 0., 2};
        std::vector<float> max_values = {1, 1, 1, 1};
        std::vector<float> decay_factors = {0, 0, 0, 0.98};
        int height = 100;
        int width = 100;
        int num_yaal = 50;
        int num_plant = 50;
        int num_steps = 1;
        Environment env(height, width, 4, decay_factors, diffusion_factors, max_values);
        env.create_yaals_and_plants(num_yaal, num_plant);
        for (int i = 0; i < num_steps; i++) {
            env.step();
        }

        // Save and load the environment
        std::string save_path = "./save/";
        ensure_directory_exists(save_path);
        save_environment(env, save_path, true);

        Environment env2 = load_environment(save_path);
        Environment env3 = load_environment(save_path);

        int num_steps2 = 100;
        for (int i = 0; i < num_steps2; i++) {
            env2.step();
        }

        //set omp num threads to 1
        omp_set_num_threads(1);
        for (int i = 0; i < num_steps2; i++) {
            env3.step();
        }

        // use is_close :
        REQUIRE(is_close(env2.map, env3.map));
    }
}

TEST_CASE("MPI_ENV") {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    Topology top = get_topology(MPI_COMM_WORLD);
    MPI_Comm solo_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_rank, 0, &solo_comm);
    SECTION("Empty env") {
        auto seed = Catch::getSeed();
        Yaal::generator.seed(seed + mpi_rank);
        YaalGenome::generator.seed(seed + mpi_rank);
        std::cout << "Hello from rank!" << mpi_rank << " of " << top.processes << " processes" << std::endl;
        std::cout << "Running with " << top.cores_per_process << " cores per process and " << top.gpus << " gpus"
                  << " with a total of " << top.gpus_memory << "MB of GPU memory" << std::endl;
        std::cout << "Running with a total of " << top.nodes << " nodes" << std::endl;
        // Divide the environment into subenvironments
        int height = 60;
        int width = 60;
        bool allow_idle = false;
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
            return;
        }
        int num_chunks = rows * columns;
        int sub_height = height / rows;
        int sub_width = width / columns;
        int row = mpi_rank / columns;
        int column = mpi_rank % columns;
        Vec2 top_left_position((float) (column * sub_width), (float) (row * sub_height));
        if (row == rows - 1) {
            sub_height += height % sub_height;
        }
        if (column == columns - 1) {
            sub_width += width % sub_width;
        }
        int num_channels = 3;
        std::vector<float> diffusion_rates = {0., 0., 0.};
        std::vector<float> max_values = {1, 1, 1};
        std::vector<float> decay_factors = {1, 1, 1};

        Environment env(sub_height, sub_width, num_channels, decay_factors, diffusion_rates, max_values,
                        std::move(top_left_position), rows, columns);
        env.real_map().chip(mpi_rank % 3, 2).setConstant(0.8);
        path save_path = "test_output/mpi_empty";
        path global_save_frame_path = save_path / "frames";
        path global_total_frame_path = save_path / "frames/total";
        path save_frame_path = save_path / "frames" / std::format("env_{}", mpi_rank);
        if (mpi_rank == 0) {
            remove_directory_recursively(save_path);
            ensure_directory_exists(save_path);
            ensure_directory_exists(global_total_frame_path);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        ensure_directory_exists(save_frame_path);
        std::string video_path = save_path / std::format("env_{}.mp4", mpi_rank);
        std::string global_video_path = save_path / "global_env.mp4";
        std::string global_total_video_path = save_path / "global_total_env.mp4";
        Stream solo_stream(video_path.c_str(), 10, cv::Size(env.map.dimension(1), env.map.dimension(0)), 1, 1, false,
                           solo_comm);
        Stream global_stream(global_video_path.c_str(), 10, cv::Size(width, height), rows, columns, false,
                             MPI_COMM_WORLD);
        auto total_offset = env.offset_padding + env.offset_sharing;
        Stream global_total_stream(global_total_video_path.c_str(), 10,
                                   cv::Size(width * 2, height * 2), rows,
                                   columns, false,
                                   MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD); // todo remove
        std::cout << "Starting steps" << std::endl;
        for (int i = 0; i < 10; i++) {
            path global_frame_path = global_save_frame_path / std::format("frame_{}.png", i);
            path global_total_frame = global_total_frame_path / std::format("frame_{}.png", i);
            path frame_path = save_frame_path / std::format("frame_{}.png", i);
            solo_stream.append_frame(env.map, frame_path.c_str());
            global_total_stream.append_frame(env.map, global_total_frame.c_str());
            Tensor<float, 3> map = env.real_map();
            std::cout << map.dimensions() << " of " << height << "," << width << std::endl;
            global_stream.append_frame(map, global_frame_path.c_str());
            if (mpi_rank == 0) {
                std::cout << "================== Step " << i << " =============" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            env.step();
        }
    }

    SECTION("Diffusion determinism") {
        int num_channels = 4;
        std::vector<float> diffusion_factors = {0., 0., 0., 1000};
        std::vector<float> max_values = {1, 1, 1, 1};
        std::vector<float> decay_factors = {0, 0, 0, 1};
        int height = 500;
        int width = 500;
        int num_steps = 10;
        Environment base_env(height, width, num_channels, decay_factors, diffusion_factors, max_values);
        base_env.mpi_world = solo_comm;
        base_env.real_map().slice(array<Index, 3>{0, 0, 3}, array<Index, 3>{100, 100, 1}).setConstant(1);
        std::cout << "Set base_env" << std::endl;
        bool allow_idle = false;
        auto [rows, columns] = grid_decomposition(top.processes, allow_idle);
        std::cout << "got grid" << std::endl;
        if (rows > columns && width > height) {
            int temp = rows;
            rows = columns;
            columns = temp;
        }
        assert(allow_idle || (columns * rows == top.processes));
        if (columns * rows <= mpi_rank) {
            std::cout << "Rank " << mpi_rank << " is idle" << std::endl;
            MPI_Finalize();
            return;
        }
        int num_chunks = rows * columns;
        int sub_height = height / rows;
        int sub_width = width / columns;
        int row = mpi_rank / columns;
        int column = mpi_rank % columns;
        Vec2 top_left_position((float) (column * sub_width), (float) (row * sub_height));
        if (row == rows - 1) {
            sub_height += height % sub_height;
        }
        if (column == columns - 1) {
            sub_width += width % sub_width;
        }
        std::cout << "Creating env" << std::endl;
        std::vector<float> diffusion_factors2 = {0., 0., 0., 2};  // to dodge the move
        Environment sub_env(sub_height, sub_width, num_channels, decay_factors, diffusion_factors2, max_values,
                            std::move(top_left_position), rows, columns);
        // Add 1 to the sub_env if needed. The concatenated map will be the same as the base_env
        std::cout << "created sub_env" << std::endl;
        Vec2 start_pos = {0.f, 0.f};
        Vec2 end_pos = {100.f, 100.f};
        if (!((end_pos.x() < sub_env.top_left_position.x() && end_pos.y() < sub_env.top_left_position.y()) ||
              (start_pos.x() > sub_env.top_left_position.x() + sub_env.width &&
               start_pos.y() > sub_env.top_left_position.y() + sub_env.height))) {
            array<Index, 3> offsets = {std::max(0l, (Index) (start_pos.y() - sub_env.top_left_position.y())),
                                       std::max(0l, (Index) (start_pos.x() - sub_env.top_left_position.x())),
                                       3};
            array<Index, 3> sizes = {
                    std::min((Index) sub_env.height - offsets[0],
                             (Index) (end_pos.y() - sub_env.top_left_position.y())),
                    std::min((Index) sub_env.width - offsets[1], (Index) (end_pos.x() - sub_env.top_left_position.x())),
                    1};
            Eigen::TensorMap<Tensor<Index, 1>> offsets_t(offsets.data(), 3);
            Eigen::TensorMap<Tensor<Index, 1>> sizes_t(sizes.data(), 3);
            std::cout << "position: " << sub_env.top_left_position << "Setting to 1 using offsets: " << offsets_t
                      << " and sizes: " << sizes_t << std::endl;
            sub_env.real_map().slice(offsets, sizes).setConstant(1);
        }
        path save_path = "test_output/diffusion_determinism";
        path total_frames_path = save_path / "frames_total";
        path frame_path = save_path / "frames";
        if (mpi_rank == 0) {
            remove_directory_recursively(save_path);
            ensure_directory_exists(frame_path);
            ensure_directory_exists(total_frames_path);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            path base_frame_path = save_path / "frames_base";
            ensure_directory_exists(base_frame_path);
            Stream base_stream((save_path / "base_env.mp4").c_str(), 10, cv::Size(width, height), 1, 1, false,
                               solo_comm);
            Stream total_base_stream((save_path / "total_base_env.mp4").c_str(), 10, cv::Size(2*width, 2*height), 1, 1, false,
                               solo_comm);
            for (int i = 0; i < num_steps; i++) {
                path base_frame_path_png = base_frame_path / std::format("frame_{}.png", i);
                path total_base_frame_path_png = base_frame_path / std::format("tframe_{}.png", i);

                Tensor<float, 3> real_map = base_env.real_map().slice(array<Index, 3>{0, 0, 1},
                                                                      array<Index, 3>{height, width, 3});
                base_stream.append_frame(real_map, base_frame_path_png.c_str());
                Tensor<float, 3> total_map = base_env.map.slice(array<Index, 3>{0, 0, 1},
                                                               array<Index, 3>{base_env.map.dimension(0),
                                                                               base_env.map.dimension(1), 3});
                total_base_stream.append_frame(total_map, total_base_frame_path_png.c_str());
                base_env.step();
            }
        } else {
            for (int i = 0; i < num_steps; i++) {
                base_env.step();
            }
        }
        std::cout << "Process " << mpi_rank << " waiting at barrier" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        path global_video_path = save_path / "global_env.mp4";
        path global_total_video_path = save_path / "global_total_env.mp4";
        Stream global_stream(global_video_path.c_str(), 10, cv::Size(width, height), rows, columns, false,
                             MPI_COMM_WORLD);
        Stream global_total_stream(global_total_video_path.c_str(), 10,
                                   cv::Size(width * 2, height * 2), rows,
                                   columns, false,
                                   MPI_COMM_WORLD);
        sub_env.mpi_sync();
        for (int i = 0; i < num_steps; i++) {
            std::cout << "Step " << i << " for process " << mpi_rank << std::endl;
            path global_frame_path = frame_path / std::format("frame_{}.png", i);
            Tensor<float, 3> real_map = sub_env.real_map().slice(array<Index, 3>{0, 0, 1},
                                                                 array<Index, 3>{sub_height, sub_width, 3});
            global_stream.append_frame(real_map, global_frame_path.c_str());
            Tensor<float, 3> total_map = sub_env.map.slice(array<Index, 3>{0, 0, 1},
                                                           array<Index, 3>{sub_env.map.dimension(0),
                                                                           sub_env.map.dimension(1), 3});
            global_total_stream.append_frame(total_map, (total_frames_path / std::format("frame_{}.png", i)).c_str());
            sub_env.step();
        }
        MPI_Barrier(MPI_COMM_WORLD);

        Tensor<float, 3> my_chunk = base_env.real_map().slice(array<Index, 3>{(Index) sub_env.top_left_position.y(),
                                                                              (Index) sub_env.top_left_position.x(), 0},
                                                              array<Index, 3>{sub_env.height, sub_env.width,
                                                                              num_channels});
        Tensor<float, 3> my_map = sub_env.real_map();
        REQUIRE(is_close(my_chunk, my_map));
    }
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size < 2) {
        SKIP("Skipping test, not < 2 processes");
    }
    boost::mpi::environment env;
    boost::mpi::communicator world;
    SECTION("SEND YAALMLP") {
        int seed = 42;
        YaalGenome::generator.seed(seed);
        YaalMLP source = YaalGenome::random(3).brain;
        if (world.rank() == 0) {
            world.send(1, 0, source.to_serialized());
        } else if (world.rank() == 1) {
            SerializedYaalMLP target_serialized;
            world.recv(0, 0, target_serialized);
            YaalMLP target = YaalMLP::from_serialized(target_serialized);
            REQUIRE(is_close(source.direction_weights, target.direction_weights));
        }

    }SECTION("SEND YaalGenome") {
        int seed = 42;
        YaalGenome::generator.seed(seed);
        YaalGenome source = YaalGenome::random(3);
        if (world.rank() == 0) {
            world.send(1, 0, source.to_serialized());
        } else if (world.rank() == 1) {
            SerializedYaalGenome target_serialized;
            world.recv(0, 0, target_serialized);
            YaalGenome target = YaalGenome::from_serialized(target_serialized);
            REQUIRE(is_close(source.brain.direction_weights, target.brain.direction_weights));
            REQUIRE(vec_is_close(source.signature, target.signature));
        }
    }SECTION("SEND YAAL") {
        int seed = 42;
        Yaal::generator.seed(seed);
        int num_channels = 4;
        Yaal source = Yaal::random(num_channels);
        if (world.rank() == 0) {
            world.send(1, 0, source.to_serialized());
        } else if (world.rank() == 1) {
            SerializedYaal target_serialized;
            world.recv(0, 0, target_serialized);
            Yaal target = Yaal::from_serialized(target_serialized);
            REQUIRE(is_close(source.body, target.body));
            REQUIRE(is_close(source.genome.brain.direction_weights, target.genome.brain.direction_weights));
            REQUIRE(vec_is_close(source.genome.signature, target.genome.signature));
            REQUIRE((source.position - target.position).cwiseAbs().isMuchSmallerThan(Constants::EPSILON));
        }
    }
}


TEST_CASE("Misc") {
    SECTION("Neighbors Indexing") {
        auto n1 = Neighbourhood::none();
        auto n2 = Neighbourhood::none();
        n1.top = 1;
        n2[0] = 1;
        n1.bottom = 2;
        n2[1] = 2;
        n1.left = 3;
        n2[2] = 3;
        n1.right = 4;
        n2[3] = 4;
        n1.top_left = 5;
        n2[4] = 5;
        n1.top_right = 6;
        n2[5] = 6;
        n1.bottom_left = 7;
        n2[6] = 7;
        n1.bottom_right = 8;
        n2[7] = 8;
        REQUIRE(n1.top == n2.top);
        REQUIRE(n1.bottom == n2.bottom);
        REQUIRE(n1.left == n2.left);
        REQUIRE(n1.right == n2.right);
        REQUIRE(n1.top_left == n2.top_left);
        REQUIRE(n1.top_right == n2.top_right);
        REQUIRE(n1.bottom_left == n2.bottom_left);
        REQUIRE(n1.bottom_right == n2.bottom_right);
        for (int i = 0; i < 8; i++) {
            REQUIRE(n1[i] == n2[i]);
        }
    }SECTION("MPI Boost send vector") {
        int comm_size;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        if (comm_size < 2) {
            SKIP("Skipping test, not < 2 processes");
        }
        boost::mpi::environment env;
        boost::mpi::communicator world;
        std::vector<int> source = {1, 2, 3, 4};
        if (world.rank() == 0) {
            world.send(1, 0, source);
        } else if (world.rank() == 1) {
            std::vector<int> target;
            world.recv(0, 0, target);
            REQUIRE(source == target);
        }
    }
}