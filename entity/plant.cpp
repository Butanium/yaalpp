#include "plant.hpp"
#include <utility>
#include <iostream>

Tensor<float, 3> Plant::default_body(int size, int num_channels) {
    Tensor<float, 3> body = Tensor<float, 3>(size, size, num_channels);
    body.setZero();
    // Fill the Green channel with ones
    body.chip(1, 2).setConstant(1);
    body.chip(num_channels - 1, 2).setConstant(1);
    float center = (float) size / 2.f - 0.5f;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float dx = (float) i - center;
            float dy = (float) j - center;
            auto slice = body.chip(i, 0).chip(j, 0);
            if (dx * dx + dy * dy > (float) (size * size) / 4.f) {
                slice.setZero();
            }
        }
    }
    return body;
}

Plant::Plant(Vec2 &&position, Tensor<float, 3> &&body) :
        position(std::move(position)),
        body(std::move(body)) {}

std::mt19937 Plant::generator = std::mt19937(std::random_device{}());

void Plant::set_random_position(const Vec2 &min, const Vec2 &max) {
    std::uniform_real_distribution<float> x_rng(min.x(), max.x());
    std::uniform_real_distribution<float> y_rng(min.y(), max.y());
    position = {x_rng(generator), y_rng(generator)};
}

Plant::Plant(int num_channels) :
        position(Vec2(0, 0)),
        body(default_body(Constants::Yaal::MIN_SIZE, num_channels)) {}
