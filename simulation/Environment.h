#pragma once

#include "../entity/Yaal.h"
#include "../utils/utils.h"
#include "../entity/plant.hpp"
#include "../diffusion/separablefilter.hpp"
#include "../topology/topology.h"

using Vec2i = Eigen::Vector2i;
using Eigen::Index;

class Environment {

/// Given a position in the environment, returns the index in the map tensor
    std::tuple<int, int> pos_to_index(const Vec2 &pos);

/// Store all the mpi receive results
    std::array<std::vector<float>, 8> mpi_receive_results =
            {{std::vector<float>(), std::vector<float>(), std::vector<float>(), std::vector<float>(),
              std::vector<float>(), std::vector<float>(), std::vector<float>(), std::vector<float>()}};
    int mpi_rank = 0;


public:
    Tensor<float, 3> map;
    const int height;
    const int width;
    const int num_channels;
    Offset offset_padding = {0, 0, 0, 0};
    Offset offset_sharing = {0, 0, 0, 0};
    const Vec2 top_left_position = Vec2::Zero();
    const SeparableFilter diffusion_filter;
    int mpi_row = 0;
    int mpi_column = 0;
    Neighbourhood neighbourhood = Neighbourhood::none();
    std::vector<Yaal> yaals = {};
    std::vector<Plant> plants = {};
    Eigen::TensorMap<Tensor<float, 3>> decay_factors;
    Eigen::TensorMap<Tensor<float, 3>> max_values;

    Environment(int height, int width, int channels,
                std::vector<float> &decay_factors_v,
                std::vector<float> &_diffusion_factor,
                std::vector<float> &max_values_v);

    Environment(int height, int width, int channels, std::vector<float> &decay_factors_v,
                std::vector<float> &_diffusion_factor, std::vector<float> &max_values_v, Vec2 &&_top_left_position,
                int num_mpi_rows, int num_mpi_columns);

    Environment(int height, int width, int channels,
                Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                Eigen::TensorMap<Tensor<float, 3>> max_values,
                const SeparableFilter &diffusion_filter,
                int offset_padding_top, int offset_padding_bottom, int offset_padding_left, int offset_padding_right,
                Vec2 top_left_position,
                std::vector<Yaal> _yaals,
                std::vector<Plant> plants_);


    Environment(Tensor<float, 3> &&map_,
                Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                Eigen::TensorMap<Tensor<float, 3>> max_values,
                const SeparableFilter &diffusion_filter,
                int offset_padding_top, int offset_padding_bottom, int offset_padding_left, int offset_padding_right,
                Vec2 top_left_position,
                std::vector<Yaal> yaals,
                std::vector<Plant> plants);


    auto get_view(const Yaal &yaal) {
        auto view_offsets = array<Index, 3>();
        auto [i, j] = pos_to_index(yaal.top_left_position());
        int fov = yaal.genome.field_of_view;
        view_offsets[0] = i - fov;
        view_offsets[1] = j - fov;
        view_offsets[2] = 0;
        auto view_dims = array<Index, 3>{(Index) 2 * fov + yaal.genome.size,
                                         (Index) 2 * fov + yaal.genome.size, (Index) num_channels};
        return map.slice(view_offsets, view_dims);
    }

    /// Add the plant body to the map
    void add_to_map(const Plant &plant);

    /// Add the yaal body to the map
    void add_to_map(const Yaal &yaal);

    /// Add a plant to the environment
    void add_plant(Plant &&plant);

    /// Add a plant to the environment
    void add_plant(const Plant &plant);

    /// Add a yaal to the environment
    void add_yaal(Yaal &&yaal);

    /// Add a yaal to the environment
    void add_yaal(const Yaal &yaal);

    /// Resolve collisions between yaals and closests, and clamp the positions inside the environment. If a Yaal is in the shared area of another MPI process, it is added to a buffer that will be sent to the other process.
    bool resolve_collisions(const std::vector<Vec2> &closests);

    /// Perform a step in the environment
    void step();

    void create_yaals_and_plants(int num_yaal, int num_plant);
};