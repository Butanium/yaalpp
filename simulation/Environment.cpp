//
// Created by clementd on 09/02/24.
//

#include <iostream>
#include <utility>
#include "Environment.h"
#include "../Constants.h"
#include "../physics/quadtree.hpp"
#include "../diffusion/separablefilter.hpp"

using Constants::Yaal::MAX_FIELD_OF_VIEW;
using Eigen::array;
using Eigen::Index;
using Constants::Environment::FILTER_SIZE;

Environment::Environment(int height, int width, int channels, std::vector<float> decay_factors_v,
                         std::vector<float> max_values_v) :
        width(width), height(height), channels(channels),
        offset_left(MAX_FIELD_OF_VIEW),
        offset_right(MAX_FIELD_OF_VIEW),
        offset_top(MAX_FIELD_OF_VIEW),
        offset_bottom(MAX_FIELD_OF_VIEW),
        top_left_position(Vec2i::Zero()),
        global_height(height),
        global_width(width),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>(decay_factors_v.data(), array<Index, 3>{1, 1, channels})),
        max_values(Eigen::TensorMap<Tensor<float, 3>>(max_values_v.data(), array<Index, 3>{1, 1, channels})),
        map(Tensor<float, 3>(width + 2 * MAX_FIELD_OF_VIEW, height + 2 * MAX_FIELD_OF_VIEW, channels)),
        diffusion_filter(SeparableFilter(FILTER_SIZE, channels, true)) {
    map.setZero();
}

Environment::Environment(Tensor<float, 3> &&map, std::vector<float> decay_factors_v,
                         std::vector<float> max_values_v, int offset_left, int offset_right, int offset_top,
                         int offset_bottom, Vec2i &&top_left_position, int global_height, int global_width)
        : map(std::move(map)), width((int) map.dimension(0)), height((int) map.dimension(1)),
          channels((int) map.dimension(2)), offset_left(offset_left), offset_right(offset_right),
          offset_top(offset_top), offset_bottom(offset_bottom), top_left_position(std::move(top_left_position)),
          decay_factors(Eigen::TensorMap<Tensor<float, 3>>(decay_factors_v.data(), array<Index, 3>{1, 1, channels})),
          max_values(Eigen::TensorMap<Tensor<float, 3>>(max_values_v.data(), array<Index, 3>{1, 1, channels})),
          diffusion_filter(SeparableFilter(FILTER_SIZE, (int) map.dimension(2), true)),
          global_height(global_height), global_width(global_width)
          {
}


Vec2i Environment::pos_to_index(const Vec2 &pos) {
    return (pos - top_left_position.cast<float>() + Vec2(offset_left, offset_top)).array().round().cast<int>();
}


void Environment::add_to_map(Yaal yaal) {
    Vec2i pos = pos_to_index(yaal.top_left_position());
    array<Index, 3> offsets = {pos.y(), pos.x(), 0};
    map.slice(offsets, yaal.genome.body.dimensions()) += yaal.genome.body;
}

void Environment::step() {
//#pragma omp parallel for schedule(static) // TODO perf: check if dynamic is useful
    for (auto &yaal: yaals) {
        auto view = get_view(yaal);
        yaal.update(view);
    }
    QuadTree quadtree(Rect(top_left_position.cast<float>(), Vec2(width, height)));
    quadtree.initialize(yaals);
    std::vector<Vec2> closests(yaals.size());
    quadtree.get_all_closest(yaals, closests);
    // TODO: Resolve collisions and clamp position to map
    map *= decay_factors.broadcast(array<int, 3>{width + 2 * MAX_FIELD_OF_VIEW, height + 2 * MAX_FIELD_OF_VIEW, 1});
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }
    diffusion_filter.apply_inplace(map);  // todo: avoid diffusing on offsets (@parakwel)
    map = map.cwiseMin(
            max_values.broadcast(array<int, 3>{width + 2 * MAX_FIELD_OF_VIEW, height + 2 * MAX_FIELD_OF_VIEW, 1}));
}



