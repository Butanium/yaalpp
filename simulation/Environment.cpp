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

Environment::Environment(int height, int width, int channels,
                         std::vector<float> &decay_factors_v,
                         std::vector<float> &diffusion_factor,
                         std::vector<float> &max_values_v) :
        width(width), height(height), channels(channels),
        offset_padding(
                {.top =  MAX_FIELD_OF_VIEW, .bottom =  MAX_FIELD_OF_VIEW, .left =  MAX_FIELD_OF_VIEW, .right =  MAX_FIELD_OF_VIEW}),
        offset_sharing({.top = 0, .bottom = 0, .left = 0, .right = 0}), // TODO: implement sharing
        top_left_position(Vec2i::Zero()),
        global_height(height),
        global_width(width),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>(decay_factors_v.data(), array<Index, 3>{1, 1, channels})),
        max_values(Eigen::TensorMap<Tensor<float, 3>>(max_values_v.data(), array<Index, 3>{1, 1, channels})),
        map(Tensor<float, 3>(height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, channels)),
        diffusion_filter(SeparableFilter(FILTER_SIZE, channels, true, std::move(diffusion_factor))) {
    map.setZero();
}

Environment::Environment(Tensor<float, 3> &&map,
                         std::vector<float> &decay_factors_v,
                         std::vector<float> &diffusion_factor,
                         std::vector<float> &max_values_v,
                         int offset_left, int offset_right, int offset_top,
                         int offset_bottom, // TODO : @clément use offset struct
                         Vec2i &&top_left_position,
                         int global_height, int global_width) :
        map(std::move(map)),
        width((int) map.dimension(0)), height((int) map.dimension(1)),
        channels((int) map.dimension(2)),
        offset_padding({.top = offset_top, .bottom = offset_bottom, .left = offset_left, .right = offset_right}),
        offset_sharing({.top = 0, .bottom = 0, .left = 0, .right = 0}), // TODO: implement sharing
        top_left_position(std::move(top_left_position)),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>(decay_factors_v.data(), array<Index, 3>{1, 1, channels})),
        max_values(Eigen::TensorMap<Tensor<float, 3>>(max_values_v.data(), array<Index, 3>{1, 1, channels})),
        diffusion_filter(SeparableFilter(FILTER_SIZE, (int) map.dimension(2), true, std::move(diffusion_factor))),
        global_height(global_height), global_width(global_width) {
}

std::tuple<int, int> Environment::pos_to_index(const Vec2 &pos) {
    Vec2 float_pos = (pos - top_left_position.cast<float>() +
                      Vec2(offset_padding.left, offset_padding.top));
    int x = (int) std::round(float_pos.x());
    int y = (int) std::round(float_pos.y());
    return {y, x};
}


void Environment::add_to_map(const Yaal &yaal) {
    auto [i, j] = pos_to_index(yaal.top_left_position());
    array<Index, 3> offsets = {i, j, 0};
    map.slice(offsets, yaal.body.dimensions()) += yaal.body;
}

bool Environment::resolve_collisions(const std::vector<Vec2> &closests) {
    // TODO : pragma omp for
    bool resolved = true;
#pragma omp parallel for schedule(static) shared(resolved)
    for (int i = 0; i < yaals.size(); i++) {
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

void Environment::step() {
    /* Evaluate the Yaals
     * Exchange the Yaals with the other processes to be able to resolve collisions
     * Resolve collisions
     * Decay and diffuse the map
     * Add the Yaals to the map
     * Exchange the Yaals that are now on the other side of the border as well as the shared map sections
     * add the yaals that crossed a border to the map
     * */
#pragma omp parallel for schedule(static) // TODO perf: check if dynamic is useful
    for (auto &yaal: yaals) {
        auto view = get_view(yaal);
        yaal.update(view);
    }
    int nb_loops = 0;
    for (int i = 0; i < 2; i++) {
        if (yaals.size() <= 1) {
            break;
        }
        nb_loops++;
        QuadTree quadtree(Rect(top_left_position.cast<float>(), Vec2(width, height)),
                          (float) yaals[0].genome.size / 2.f);
        quadtree.initialize(yaals);
        std::vector<Vec2> closests(yaals.size());
        quadtree.get_all_closest(yaals, closests);
        if (resolve_collisions(closests)) {
            break;
        }
    }
    // TODO?: put this in diffusion filter to parallelize it
    map *= decay_factors.broadcast(array<int, 3>{height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, 1});
    diffusion_filter.apply_inplace(map, offset_padding + offset_sharing); // TODO : check sharing offset
#pragma omp parallel for schedule(static) //TODO?: c'est des carrés et quand ça cogne ça fait pas beau
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }
    map = map.cwiseMin(
            max_values.broadcast(array<int, 3>{height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, 1}));
}

void Environment::add_yaal(Yaal &&yaal) {
    yaals.push_back(std::move(yaal));
}

void Environment::add_yaal(const Yaal &yaal) {
    yaals.push_back(yaal);
}
