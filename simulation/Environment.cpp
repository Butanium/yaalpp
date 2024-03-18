#include <iostream>
#include <utility>
#include "Environment.h"
#include "../physics/quadtree.hpp"
#include <set>
using Constants::Yaal::MAX_FIELD_OF_VIEW;
using Constants::Yaal::MAX_SIZE;
using Eigen::array;
using Eigen::Index;
using Constants::Environment::FILTER_SIZE;
using Constants::Environment::SHARED_SIZE;
using Constants::MPI::MAP_TAG;
using Constants::MPI::YAAL_TAG;

Environment::Environment(int height, int width, int channels,
                         std::vector<float> &decay_factors_v,
                         std::vector<float> &_diffusion_factor,
                         std::vector<float> &max_values_v) :
        map(Tensor<float, 3>(height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, channels)), height(height),
        width(width),
        num_channels(channels),
        offset_padding(
                {.top =  MAX_FIELD_OF_VIEW, .bottom =  MAX_FIELD_OF_VIEW, .left =  MAX_FIELD_OF_VIEW, .right =  MAX_FIELD_OF_VIEW}),
        diffusion_filter(SeparableFilter(FILTER_SIZE, channels, true, std::move(_diffusion_factor))),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>

                              (decay_factors_v.

                                      data(), array<Index, 3>{1, 1, channels}

                              )),
        max_values(Eigen::TensorMap<Tensor<float, 3>>
                           (max_values_v.

                                   data(), array<Index, 3>{1, 1, channels}

                           )) {
    map.

            setZero();

}

Environment::Environment(int height, int width,
                         int channels,
                         std::vector<float> &decay_factors_v,
                         std::vector<float> &_diffusion_factor,
                         std::vector<float> &max_values_v,
                         Vec2 &&_top_left_position,
                         int num_mpi_rows, int num_mpi_columns) :
        height(height),
        width(width),
        num_channels(channels),
        top_left_position(std::move(_top_left_position)),
        diffusion_filter(SeparableFilter(FILTER_SIZE, channels, true, std::move(_diffusion_factor))),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>

                              (decay_factors_v.

                                      data(), array<Index, 3>{1, 1, channels}

                              )),
        max_values(Eigen::TensorMap<Tensor<float, 3>>
                           (max_values_v.

                                   data(), array<Index, 3>{1, 1, channels}

                           )) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank
    );
    mpi_row = mpi_rank / num_mpi_columns;
    mpi_column = mpi_rank % num_mpi_columns;
//@formatter:off
    const std::array<int, 8> sharing_sizes = {{
        (int) SHARED_SIZE * width * num_channels,// top
        (int) SHARED_SIZE * width * num_channels,// bottom
        (int) height * SHARED_SIZE * num_channels,// left
        (int) height * SHARED_SIZE * num_channels,// right
        (int) SHARED_SIZE * SHARED_SIZE * num_channels,// top_left
        (int) SHARED_SIZE * SHARED_SIZE * num_channels,// top_right
        (int) SHARED_SIZE * SHARED_SIZE * num_channels,// bottom_left
        (int) SHARED_SIZE * SHARED_SIZE * num_channels,// bottom_right
    }};
    //@formatter:on
    for (
            int i = 0;
            i < 8; i++) {
        if (neighbourhood.
                add(i, mpi_rank, mpi_row, mpi_column, num_mpi_rows, num_mpi_columns
        )) {
//            mpi_receive_results[i] = new float[sharing_sizes[i]];
        }
    }
    for (
            int i = 0;
            i < 4; i++) {
        if (neighbourhood[i] != MPI_PROC_NULL) {
            offset_sharing[i] =
                    SHARED_SIZE;
        } else {
            offset_padding[i] =
                    MAX_FIELD_OF_VIEW;
        }
    }

    map = Tensor<float, 3>(
            height + offset_padding.top + offset_padding.bottom + offset_sharing.top + offset_sharing.bottom,
            width + offset_padding.left + offset_padding.right + offset_sharing.left + offset_sharing.right,
            channels);
    map.

            setZero();

    std::cout << "\nSubenvironment " << mpi_rank << " at r,c:" << mpi_row << "," << mpi_column << "position: "
              << top_left_position[0] << ", " << top_left_position[1] << " with size " << height << " "
              << width << "\nNeighborhood: " << neighbourhood << "\nSharing offset" << offset_sharing
              << "\nPadding offset"
              << offset_padding <<
              std::endl;
}

Environment::Environment(int height, int width, int channels, Eigen::TensorMap<Tensor<float, 3>>

decay_factors,
                         Eigen::TensorMap<Tensor<float, 3>>
                         max_values,
                         const SeparableFilter &diffusion_filter,
                         int offset_padding_top,
                         int offset_padding_bottom,
                         int offset_padding_left,
                         int offset_padding_right, Vec2
                         top_left_position,
                         std::vector<Yaal> _yaals,
                         std::vector<Plant>
                         plants_) :

        height(height),
        width(width),
        num_channels(channels),
        offset_padding(
                {.top = offset_padding_top, .bottom = offset_padding_bottom, .left = offset_padding_left, .right = offset_padding_right}),
        top_left_position(std::move(top_left_position)),
        diffusion_filter(diffusion_filter),
        yaals(std::move(_yaals)),
        plants(std::move(plants_)), decay_factors(decay_factors),
        max_values(max_values) {
    map = Tensor<float, 3>(height + offset_padding.vertical(), width + offset_padding.horizontal(), channels);
    map.setZero();
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }
    for (auto &plant: plants) {
        add_to_map(plant);
    }
}

Environment::Environment(Tensor<float, 3> &&map_,
                         Eigen::TensorMap<Tensor<float, 3>>

                         decay_factors,
                         Eigen::TensorMap<Tensor<float, 3>>
                         max_values,
                         const SeparableFilter &diffusion_filter,
                         int offset_padding_top,
                         int offset_padding_bottom,
                         int offset_padding_left,
                         int offset_padding_right,
                         Vec2
                         top_left_position,
                         std::vector<Yaal> yaals,
                         std::vector<Plant>
                         plants) :

        map(std::move(map_)),
        height((int) map.dimension(0) - 2 * MAX_FIELD_OF_VIEW),
        width((int) map.dimension(1) - 2 * MAX_FIELD_OF_VIEW),
        num_channels((int) map.dimension(2)),
        offset_padding(
                {.top = offset_padding_top, .bottom = offset_padding_bottom, .left = offset_padding_left, .right = offset_padding_right}),
        top_left_position(std::move(top_left_position)),
        diffusion_filter(diffusion_filter),
        yaals(std::move(yaals)), plants(std::move(plants)),
        decay_factors(decay_factors),
        max_values(max_values) {
}


void Environment::add_plant(Plant &&plant) {
    add_to_map(plant);
    plants.push_back(std::move(plant));
}

void Environment::add_plant(const Plant &plant) {
    plants.push_back(plant);
    add_to_map(plant);
}

void Environment::add_yaal(Yaal &&yaal) {
    add_to_map(yaal);
    yaals.push_back(std::move(yaal));
}

void Environment::add_yaal(const Yaal &yaal) {
    yaals.push_back(yaal);
    add_to_map(yaal);
}

void Environment::create_yaals_and_plants(int num_yaal, int num_plant) {
    for (int i = 0; i < num_yaal; i++) {
        Yaal yaal = Yaal::random(num_channels);
        yaal.set_random_position(Vec2((float) MAX_SIZE / 2., (float) MAX_SIZE / 2.),
                                 Vec2(width - MAX_SIZE / 2., height - MAX_SIZE / 2.));
        yaal.position += top_left_position;
        add_yaal(yaal);
    }
    for (int i = 0; i < num_plant; i++) {
        Plant plant = Plant(num_channels);
        plant.set_random_position(Vec2((float) MAX_SIZE / 2., (float) MAX_SIZE / 2.),
                                  Vec2(width - MAX_SIZE / 2., height - MAX_SIZE / 2.));
        plant.position += top_left_position;
        add_plant(plant);
    }
}

std::tuple<int, int> Environment::pos_to_index(const Vec2 &pos) {
    Vec2 float_pos = (pos - top_left_position +
                      Vec2(offset_padding.left, offset_padding.top));
    int x = (int) std::round(float_pos.x());
    int y = (int) std::round(float_pos.y());
    return {y, x};
}

void Environment::add_to_map(const Plant &plant) {
    auto [i, j] = pos_to_index(
            plant.position - Vec2((float) plant.body.dimension(0) / 2.f, (float) plant.body.dimension(1) / 2.f));
    array<Index, 3> offsets = {i, j, 0};
    auto slice = map.slice(offsets, plant.body.dimensions());
#pragma omp critical
    {
        slice += plant.body;
    }
}

void Environment::add_to_map(const Yaal &yaal) {
    auto [i, j] = pos_to_index(yaal.top_left_position());
    array<Index, 3> offsets = {i, j, 0};
    auto slice = map.slice(offsets, yaal.body.dimensions());
#pragma omp critical
    {
        slice += yaal.body;
    };
}

bool Environment::resolve_collisions(const std::vector<Vec2> &closests) {
    bool resolved = true;
#pragma omp parallel for schedule(static) shared(resolved)
    for (int i = 0; i < (int) yaals.size(); i++) {
        // resolve potential collision with other yaal
        Vec2 diff = closests[i] - yaals[i].position;
        float size = (float) yaals[i].genome.size / 2.f;
        float overlap = 2 * size - diff.norm();
        // TODO : when the sizes are not the same, put Yaals in quadtree, not only positions
        if (overlap > Constants::PHYSICS_EPSILON) {
            resolved = false;
            diff *= overlap / diff.norm();
            yaals[i].position -= diff / 2.f;
        }
        if (yaals[i].position.x() < size || yaals[i].position.y() < size ||
            yaals[i].position.x() > (float) width - size || yaals[i].position.y() > (float) height - size) {
            resolved = false;
            yaals[i].position = yaals[i].position.cwiseMax(Vec2(size, size)).cwiseMin(
                    Vec2((float) width - size, (float) height - size));
        }
    }
    return resolved;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedValue"

void Environment::mpi_sync() {
    // Send and receive map chunks to neighbours
    auto tot_offset = offset_padding + offset_sharing;
    auto recv_requests = std::vector<MPI_Request>();
    recv_requests.reserve(8);
    // @formatter:off
    const std::array<std::array<Index, 3>, 8> send_offsets = {{
        {{tot_offset.top, tot_offset.left, 0}},                          // top
        {{tot_offset.top + height - SHARED_SIZE, tot_offset.left, 0}},   // bottom
        {{tot_offset.top, tot_offset.left, 0}},                          // left
        {{tot_offset.top, tot_offset.left + width - SHARED_SIZE, 0}},    // right
        {{tot_offset.top, tot_offset.left, 0}},                          // top_left
        {{tot_offset.top, tot_offset.left + width - SHARED_SIZE, 0}},    // top_right
        {{tot_offset.top + height - SHARED_SIZE, tot_offset.left, 0}},   // bottom_left
        {{tot_offset.top + height - SHARED_SIZE, tot_offset.left + width - SHARED_SIZE, 0}} // bottom_right
    }};

    const std::array<std::array<Index, 3>, 8> recv_offsets = {{
        {{0, tot_offset.left, 0}},                                // top
        {{height + tot_offset.top, tot_offset.left, 0}},          // bottom
        {{tot_offset.top, 0, 0}},                                 // left
        {{tot_offset.top, tot_offset.left + width, 0}},           // right
        {{0, 0, 0}},                                              // top_left
        {{0, tot_offset.left + width, 0}},                        // top_right
        {{tot_offset.top + height, 0, 0}},                        // bottom_left
        {{tot_offset.top + height, tot_offset.left + width, 0}}   // bottom_right
    }};

    const std::array<std::array<Index, 3>, 8> share_dims = {{
        {{SHARED_SIZE, width, num_channels}},                  // top
        {{SHARED_SIZE, width, num_channels}},                  // bottom
        {{height, SHARED_SIZE, num_channels}},                 // left
        {{height, SHARED_SIZE, num_channels}},                 // right
        {{SHARED_SIZE, SHARED_SIZE, num_channels}},            // top_left
        {{SHARED_SIZE, SHARED_SIZE, num_channels}},            // top_right
        {{SHARED_SIZE, SHARED_SIZE, num_channels}},            // bottom_left
        {{SHARED_SIZE, SHARED_SIZE, num_channels}}             // bottom_right
    }};
    float** mpi_receive_results = new float*[8];
    // @formatter:on
    int to_recv = 0;
    for (int i = 0; i < 8; i++) {
        int rank = neighbourhood[i];
        if (rank == MPI_PROC_NULL) {
            recv_requests[i] = MPI_REQUEST_NULL;
            continue;
        }
        to_recv++;
        Tensor<float, 3> neighbor_map = map.slice(send_offsets[i], share_dims[i]);
        Eigen::TensorMap<Tensor<float, 3>> t_map(neighbor_map.data(), share_dims[i]);
        Tensor<bool, 0> test1 = ((neighbor_map.chip(mpi_rank % 3, 2) - 0.8f).abs() < Constants::EPSILON).all();
        Tensor<bool, 0> test2 = (neighbor_map.chip((mpi_rank+ 1) % 3, 2) < Constants::EPSILON).all();
        Tensor<bool, 0> test3 = (neighbor_map.chip((mpi_rank+ 2) % 3, 2) < Constants::EPSILON ).all();
        if (!test1(0)) { std::cerr << "No only 1s" << std::endl; }
        if (!test2(0)) { std::cerr << "No only 0 in channel +1" << std::endl; }
        if (!test3(0)) { std::cerr << "No only 0 in channel +2" << std::endl; }
        test1 = ((t_map.chip(mpi_rank % 3, 2) - 0.8f).abs() < Constants::EPSILON).all();
        test2 = (t_map.chip((mpi_rank + 1) % 3, 2) < Constants::EPSILON).all();
        test3 = (t_map.chip((mpi_rank + 2) % 3, 2) < Constants::EPSILON ).all();
        if (!test1(0)) { std::cerr << "No only 1s in tmap" << std::endl; }
        if (!test2(0)) { std::cerr << "No only 0 in channel +1 in tmap" << std::endl; }
        if (!test3(0)) { std::cerr << "No only 0 in channel +2 in tmap" << std::endl; }

        auto shared_size = share_dims[i][0] * share_dims[i][1] * share_dims[i][2];
        mpi_receive_results[i] = new float[shared_size];
        assert((int) neighbor_map.size() == shared_size);
        MPI_Request send_request;
        MPI_Isend(neighbor_map.data(), (int) shared_size, MPI_FLOAT, rank, MAP_TAG, MPI_COMM_WORLD,
                  &send_request);
        MPI_Irecv(mpi_receive_results[i], (int) shared_size, MPI_FLOAT, rank, MAP_TAG, MPI_COMM_WORLD,
                  &recv_requests[i]);

    }
    std::cout << "Process " << mpi_rank << " waiting for " << to_recv << " map chunks\n";
    // When a receive is done, add the data to the map
    // To do that we create an omp task per receive request
#pragma omp parallel shared(recv_requests, mpi_receive_results, recv_offsets, share_dims, to_recv, std::cout, map, mpi_rank, neighbourhood, std::cerr)
    {
#pragma omp single nowait
        {
            while (to_recv > 0) {
                int completed_idx;
                // TODO: check that test any won't return the same index twice
                MPI_Waitany(8, recv_requests.data(), &completed_idx, MPI_STATUS_IGNORE);
                if (completed_idx != MPI_UNDEFINED) {
                    std::cout << "Process " << mpi_rank << " received map from " << neighbourhood[completed_idx]
                              << "\n";
                    to_recv--;
                    auto dims_ = share_dims[completed_idx];
                    auto data_ = mpi_receive_results[completed_idx];
                    std::set<float> values;
                    for (int i =0; i < dims_[0] * dims_[1] * dims_[2]; i++) {
//                        values.insert(data_[i]);
                        std::cout << data_[i] << " ";
                    }
//                    for (float v: values) {
//                        std::cout << v << " ";
//                    }
                    std::cout << std::endl;

#pragma omp task default(none) shared(map, mpi_receive_results, recv_offsets, share_dims, neighbourhood, std::cerr) firstprivate(completed_idx)
                    {
                        auto offset = recv_offsets[completed_idx];
                        auto dims = share_dims[completed_idx];
                        auto data = mpi_receive_results[completed_idx];
                        auto neighbor_map = Eigen::TensorMap<Tensor<float, 3>>
                                (data, dims);
                        Tensor<bool, 0> test1 = ((neighbor_map.chip(neighbourhood[completed_idx] % 3, 2) - 0.8f).abs() < Constants::EPSILON).all();
                        Tensor<bool, 0> test2 = (neighbor_map.chip((neighbourhood[completed_idx] + 1) % 3, 2) < Constants::EPSILON).all();
                        Tensor<bool, 0> test3 = (neighbor_map.chip((neighbourhood[completed_idx] + 2) % 3, 2) < Constants::EPSILON ).all();
                        if (!test1(0)) { std::cerr << "No only 1s in rcv" << std::endl; }
                        if (!test2(0)) { std::cerr << "No only 0 in channel +1 in rcv" << std::endl; }
                        if (!test3(0)) { std::cerr << "No only 0 in channel +2 in rcv" << std::endl; }
//                        map.slice(offset, dims) = neighbor_map;
                        Tensor<float, 3> dummy_t(dims);
                        dummy_t.setZero();
                        dummy_t.chip(neighbourhood[completed_idx] % 3, 2).setConstant(0.8f);
                        map.slice(offset, dims) = dummy_t;

                    }
                } else {
                    std::cout << "Process " << mpi_rank << " received MPI_UNDEFINED" << std::endl;
                    throw std::runtime_error("MPI_UNDEFINED received from waitany");
                }
            }
        }
    }
    std::cout << "Process " << mpi_rank << " finished waiting for map chunks" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

}

#pragma clang diagnostic pop

void Environment::step() {
    /* Evaluate the Yaals
     * Exchange the Yaals with the other processes to be able to resolve collisions
     * Resolve collisions
     * Decay and diffuse the map
     * Add the Yaals to the map
     * Exchange the Yaals that are now on the other side of the border as well as the shared map sections
     * add the yaals that crossed a border to the map
     * */
    auto tot_offset = offset_padding + offset_sharing;
#pragma omp parallel for schedule(static) // TODO perf: check if dynamic is useful
    for (auto &yaal: yaals) {
        // todo remove
        Vec2 relative_pos = yaal.position - top_left_position;
        if (relative_pos.x() < MAX_SIZE / 2.f || relative_pos.y() < MAX_SIZE / 2.f ||
            relative_pos.x() > width - MAX_SIZE / 2.f || relative_pos.y() > height - MAX_SIZE / 2.f) {
            std::cout << "Skipping Yaal: " << yaal.position << " is out of sub env bounds" << std::endl;
            continue;
        }
        auto view = get_view(yaal);
        yaal.update(view);
    }

    // Solve collisions
    // TODO : if all the rest is done on GPU, reslove them with glouton n^2 on GPU, not quadtree on CPU
    for (int i = 0; i < 2; i++) {
        if (yaals.size() + plants.size() <= 1 || yaals.empty()) {
            break;
        }

        // initialize quadtree
        QuadTree quadtree(Rect(top_left_position, Vec2(width, height)),
                          (float) MAX_SIZE / 2.f);
        quadtree.initialize(yaals);
        quadtree.add_plants(plants);

        // get closest yaals and resolve collisions
        std::vector<Vec2> closests(yaals.size());
        quadtree.get_all_closest(yaals, closests);
        if (resolve_collisions(closests)) {
            break;
        }
    }

    // TODO?: put this in diffusion filter to parallelize it
    // TODO?: use slicing to avoid useless operation on border
    map *= decay_factors.broadcast(array<int, 3>{height + tot_offset.vertical(),
                                                 width + tot_offset.horizontal(), 1});
    diffusion_filter.apply_inplace(map, tot_offset);

#pragma omp parallel for schedule(static)
    for (auto &yaal: yaals) {
        // TODO: remove this check for yaal in sub_env boudn and implement yaal passing
        Vec2 relative_pos = yaal.position - top_left_position;
        if (relative_pos.x() < MAX_SIZE / 2.f || relative_pos.y() < MAX_SIZE / 2.f ||
            relative_pos.x() > width - MAX_SIZE / 2.f || relative_pos.y() > height - MAX_SIZE / 2.f) {
            std::cout << "Skipping Yaal: " << yaal.position << " is out of sub env bounds" << std::endl;
            continue;
        }
        add_to_map(yaal);
    }
#pragma omp parallel for schedule(static)
    for (auto &plant: plants) {
        add_to_map(plant);
    }

    // TODO?: parallelize it
    // TODO?: use slicing to avoid useless operation on border
    map = map.cwiseMin(
            max_values.broadcast(array<int, 3>{height + tot_offset.vertical(),
                                               width + tot_offset.horizontal(), 1}));
    mpi_sync();
};