//
// Created by clementd on 09/02/24.
//

#include <iostream>
#include "Environment.h"
#include "../Constants.h"

using Constants::Yaal::MAX_FIELD_OF_VIEW;
using Eigen::array;
using Eigen::Index;

Environment::Environment(int width, int height, int channels) : width(width), height(height), channels(channels),
                                                                offset_left(MAX_FIELD_OF_VIEW),
                                                                offset_right(MAX_FIELD_OF_VIEW),
                                                                offset_top(MAX_FIELD_OF_VIEW),
                                                                offset_bottom(MAX_FIELD_OF_VIEW),
                                                                top_left_position(Vec2i::Zero()) {
    map = Tensor<float, 3>(width + 2 * MAX_FIELD_OF_VIEW, height + 2 * MAX_FIELD_OF_VIEW, channels);
    map.setZero();
}

Environment::Environment(Tensor<float, 3> &&map, int offset_left, int offset_right, int offset_top,
                         int offset_bottom, Vec2i top_left_position) : map(std::move(map)),
                                                                       width((int) map.dimension(0)),
                                                                       height((int) map.dimension(1)),
                                                                       channels((int) map.dimension(2)),
                                                                       offset_left(offset_left),
                                                                       offset_right(offset_right),
                                                                       offset_top(offset_top),
                                                                       offset_bottom(offset_bottom),
                                                                       top_left_position(top_left_position) {}


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
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }

}



