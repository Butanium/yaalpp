#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Constants.h"

using Eigen::Tensor;
using Vec2 = Eigen::Vector2f;

Tensor<float, 3> default_body(int size, int num_channels);

class Plant{
public:
    Vec2 position;
    static const Tensor<float, 3> DEFAULT_BODY;
    Tensor<float, 3> body;

    Plant(Vec2 &&position, Tensor<float, 3> &&body);

    static Plant random_plant(int height, int width);

    static Plant random_plant(int num_channels, int height, int width);
};