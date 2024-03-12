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

Plant Plant::random_plant(int num_channels, int height, int width) {
    // Create a body with a green circle in the middle
    Tensor<float, 3> body = default_body(Constants::Yaal::MIN_SIZE, num_channels);

    // Random position
    std::uniform_real_distribution<float> x_rng((float) Constants::Yaal::MAX_FIELD_OF_VIEW,
                                                (float) width - Constants::Yaal::MAX_FIELD_OF_VIEW);
    std::uniform_real_distribution<float> y_rng((float) Constants::Yaal::MAX_FIELD_OF_VIEW,
                                                (float) height - Constants::Yaal::MAX_FIELD_OF_VIEW);

    float x = x_rng(Plant::generator);
    float y = y_rng(Plant::generator);
    Vec2 pos = {x, y};
    return {std::move(pos), std::move(body)};
}
