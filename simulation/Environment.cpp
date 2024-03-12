//
// Created by clementd on 09/02/24.
//
#pragma once

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

Environment::Environment(int height, int width, int channels, Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                         Eigen::TensorMap<Tensor<float, 3>> max_values, const SeparableFilter &diffusion_filter,
                         int offset_padding_top, int offset_padding_bottom, int offset_padding_left,
                         int offset_padding_right, int offset_sharing_top, int offset_sharing_bottom,
                         int offset_sharing_left, int offset_sharing_right, Vec2i top_left_position, int global_height,
                         int global_width, std::vector<Yaal> yaals, std::vector<Plant> plants) :
        width(width), height(height), channels(channels),
        offset_padding({.top = offset_padding_top, .bottom = offset_padding_bottom, .left = offset_padding_left, .right = offset_padding_right}),
        offset_sharing({.top = offset_sharing_top, .bottom = offset_sharing_bottom, .left = offset_sharing_left, .right = offset_sharing_right}),
        top_left_position(std::move(top_left_position)),
        global_height(global_height), global_width(global_width),
        decay_factors(decay_factors),
        max_values(max_values),
        diffusion_filter(diffusion_filter),
        yaals(std::move(yaals)), plants(std::move(plants)),
        // TODO : shouldn't this be height + offset + offset + offset + offset, width + ... ?
        map(Tensor<float, 3>(height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, channels)) {
    map.setZero();
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }
    for (auto &plant: plants) {
        add_to_map(plant);
    }
}

Environment::Environment(Tensor<float, 3> &&map_,
                         Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                         Eigen::TensorMap<Tensor<float, 3>> max_values,
                         const SeparableFilter& diffusion_filter,
                         int offset_padding_top, int offset_padding_bottom, int offset_padding_left, int offset_padding_right,
                         int offset_sharing_top, int offset_sharing_bottom, int offset_sharing_left, int offset_sharing_right,
                         Vec2i top_left_position,
                         int global_height, int global_width,
                         std::vector<Yaal> yaals,
                         std::vector<Plant> plants) :
        map(std::move(map_)),
        height((int) map.dimension(0) - 2 * MAX_FIELD_OF_VIEW),
        width((int) map.dimension(1) - 2 * MAX_FIELD_OF_VIEW),
        global_height(global_height), global_width(global_width),
        channels((int) map.dimension(2)),
        offset_padding({.top = offset_padding_top, .bottom = offset_padding_bottom, .left = offset_padding_left, .right = offset_padding_right}),
        offset_sharing({.top = offset_sharing_top, .bottom = offset_sharing_bottom, .left = offset_sharing_left, .right = offset_sharing_right}),
        top_left_position(std::move(top_left_position)),
        diffusion_filter(diffusion_filter),
        yaals(std::move(yaals)), plants(std::move(plants)),
        decay_factors(decay_factors),
        max_values(max_values) {
}

std::tuple<int, int> Environment::pos_to_index(const Vec2 &pos) {
    Vec2 float_pos = (pos - top_left_position.cast<float>() +
                      Vec2(offset_padding.left, offset_padding.top));
    int x = (int) std::round(float_pos.x());
    int y = (int) std::round(float_pos.y());
    return {y, x};
}

void Environment::add_to_map(const Plant &plant) {
    //topleftposition : position - Vec2((float) genome.size / 2.f, (float) genome.size / 2.f)
    auto [i, j] = pos_to_index(plant.position - Vec2((float) plant.body.dimension(0) / 2.f, (float) plant.body.dimension(1) / 2.f));
    array<Index, 3> offsets = {i, j, 0};
    map.slice(offsets, plant.body.dimensions()) += plant.body;
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

    // Solve collisions
    // TODO : if all the rest is done on GPU, reslove them with glouton n^2 on GPU, not quadtree on CPU
    for (int i = 0; i < 2; i++) {
        if (yaals.size() + plants.size() <= 1) {
            break;
        }

        // initialize quadtree
        QuadTree quadtree(Rect(top_left_position.cast<float>(), Vec2(width, height)),
                          (float) yaals[0].genome.size / 2.f);
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
    map *= decay_factors.broadcast(array<int, 3>{height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, 1});
    diffusion_filter.apply_inplace(map, offset_padding + offset_sharing); // TODO : check sharing offset

#pragma omp parallel for schedule(static) //TODO?: c'est des carrés et quand ça cogne ça fait pas beau
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }
#pragma omp parallel for schedule(static) // TODO perf: check if dynamic is useful
    for (auto &plant: plants) {
        add_to_map(plant);
    }

    map = map.cwiseMin(
            max_values.broadcast(array<int, 3>{height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, 1}));
}

void Environment::add_plant(Plant &&plant) {
    plants.push_back(std::move(plant));
}

void Environment::add_plant(const Plant &plant) {
    plants.push_back(plant);
}

void Environment::add_yaal(Yaal &&yaal) {
    yaals.push_back(std::move(yaal));
}

void Environment::add_yaal(const Yaal &yaal) {
    yaals.push_back(yaal);
}
