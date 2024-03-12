#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Constants.h"

using Eigen::Tensor;
using Vec2 = Eigen::Vector2f;


class Plant {
    static Tensor<float, 3> default_body(int size, int num_channels);

public:
    static std::mt19937 generator;
    Vec2 position;
    Tensor<float, 3> body;

    Plant(Vec2 &&position, Tensor<float, 3> &&body);

    void set_random_position(const Vec2 &min, const Vec2 &max);

    Plant(int num_channels);
};