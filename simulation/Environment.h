//
// Created by clementd on 09/02/24.
//

#ifndef YAALPP_ENVIRONMENT_H
#define YAALPP_ENVIRONMENT_H

#include "../entity/Yaal.h"
#include "../entity/plant.hpp"
#include "../diffusion/separablefilter.hpp"

using Vec2i = Eigen::Vector2i;
using Eigen::Index;

class Environment {
    /// Given a position in the environment, returns the index in the map tensor
    std::tuple<int, int> pos_to_index(const Vec2 &pos);

public:
    Tensor<float, 3> map;
    const int height;
    const int width;
    const int global_height;
    const int global_width;
    const int channels;
    const Offset offset_padding;
    const Offset offset_sharing;
    const Vec2i top_left_position;
    const SeparableFilter diffusion_filter;
    std::vector<Yaal> yaals = {};
    std::vector<Plant> plants = {};
    Eigen::TensorMap<Tensor<float, 3>> decay_factors;
    Eigen::TensorMap<Tensor<float, 3>> max_values;

    Environment(int height, int width, int channels,
                std::vector<float> &decay_factors_v,
                std::vector<float> &diffusion_factor,
                std::vector<float> &max_values_v);

    Environment(int height, int width, int channels,
                Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                Eigen::TensorMap<Tensor<float, 3>> max_values,
                const SeparableFilter& diffusion_filter,
                int offset_padding_top, int offset_padding_bottom, int offset_padding_left, int offset_padding_right,
                int offset_sharing_top, int offset_sharing_bottom, int offset_sharing_left, int offset_sharing_right,
                Vec2i top_left_position,
                int global_height, int global_width,
                std::vector<Yaal> yaals,
                std::vector<Plant> plants);


    Environment(Tensor<float, 3> &&map_,
                Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                Eigen::TensorMap<Tensor<float, 3>> max_values,
                const SeparableFilter& diffusion_filter,
                int offset_padding_top, int offset_padding_bottom, int offset_padding_left, int offset_padding_right,
                int offset_sharing_top, int offset_sharing_bottom, int offset_sharing_left, int offset_sharing_right,
                Vec2i top_left_position,
                int global_height, int global_width,
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
                                         (Index) 2 * fov + yaal.genome.size, (Index) channels};
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
};


#endif //YAALPP_ENVIRONMENT_H
