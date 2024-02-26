//
// Created by clementd on 09/02/24.
//

#ifndef YAALPP_ENVIRONMENT_H
#define YAALPP_ENVIRONMENT_H

#include "../entity/Yaal.h"
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
    const int offset_left;
    const int offset_right;
    const int offset_top;
    const int offset_bottom;
    const Vec2i top_left_position;
    const SeparableFilter diffusion_filter;
    std::vector<Yaal> yaals = {};
    Eigen::TensorMap<Tensor<float, 3>> decay_factors;
    Eigen::TensorMap<Tensor<float, 3>> max_values;

    Environment(int width, int height, int channels, std::vector<float> decay_factors_v,
                std::vector<float> max_values_v);

    Environment(Tensor<float, 3> &&map, std::vector<float> decay_factors, std::vector<float> max_values,
                int offset_left, int offset_right, int offset_top, int offset_bottom,
                Vec2i &&top_left_position, int global_height, int global_width);

    auto get_view(const Yaal &yaal) {
        auto view_offsets = array<Index, 3>();
        auto [i, j] = pos_to_index(yaal.top_left_position());
        int fov = yaal.genome.field_of_view;
        view_offsets[1] = i - fov;
        view_offsets[0] = j - fov;
        view_offsets[2] = 0;
        auto view_dims = array<Index, 3>{(Index) 2 * fov + yaal.genome.size,
                                         (Index) 2 * fov + yaal.genome.size, (Index) channels};
        return map.slice(view_offsets, view_dims);
    }

    /// Add the yaal to the environment
    void add_to_map(const Yaal &yaal);

    /// Perform a step in the environment
    void step();

};


#endif //YAALPP_ENVIRONMENT_H
