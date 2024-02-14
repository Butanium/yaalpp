//
// Created by clementd on 09/02/24.
//

#ifndef YAALPP_ENVIRONMENT_H
#define YAALPP_ENVIRONMENT_H

#include "../entity/Yaal.h"

using Vec2i = Eigen::Vector2i;


class Environment {
    /// Given a position in the environment, returns the index in the map tensor
    Vec2i pos_to_index(const Vec2 &pos);

public:
    Tensor<float, 3> map;
    const int width;
    const int height;
    const int channels;
    const int offset_left;
    const int offset_right;
    const int offset_top;
    const int offset_bottom;
    const Vec2i top_left_position;
    std::vector<Yaal> yaals = {};

    Environment(int width, int height, int channels);

    Environment(Tensor<float, 3> &&map, int offset_left, int offset_right, int offset_top, int offset_bottom,
                Vec2i top_left_position);

    auto get_view(const Yaal &yaal);

    /// Add the yaal to the environment
    void add_to_map(Yaal yaal);

    /// Perform a step in the environment
    void step();

};


#endif //YAALPP_ENVIRONMENT_H
