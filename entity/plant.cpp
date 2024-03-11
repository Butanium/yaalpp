#include "plant.hpp"
#include <utility>
#include <iostream>

Tensor<float, 3> default_body(int size, int num_channels) {
    Tensor<float, 3> body = Tensor<float, 3>(size, size, num_channels);
    body.setZero();
    // Fill the Green channel with ones
    body.chip(1, 2).setConstant(1);
    body.chip(num_channels-1, 2).setConstant(1);
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
    body(std::move(body))
{}

const Tensor<float, 3> Plant::DEFAULT_BODY = default_body(Constants::Yaal::MIN_SIZE, Constants::Environment::NUM_CHANNELS);

Plant Plant::random_plant(int height, int width) {
    // Create a body with a green circle in the middle
    Tensor<float, 3> body = DEFAULT_BODY;

    // Random position
    // TODO : fix parallel random
    float x = (float) (rand() % (width - 2 * Constants::Yaal::MAX_FIELD_OF_VIEW)) + Constants::Yaal::MAX_FIELD_OF_VIEW;
    float y = (float) (rand() % (height - 2 * Constants::Yaal::MAX_FIELD_OF_VIEW)) + Constants::Yaal::MAX_FIELD_OF_VIEW;
    Vec2 pos = {x, y};
    return Plant(std::move(pos), std::move(body));
}

Plant Plant::random_plant(int num_channels, int height, int width) {
    // Create a body with a green circle in the middle
    Tensor<float, 3> body = default_body(Constants::Yaal::MIN_SIZE, num_channels);

    // Random position
    // TODO : fix parallel random
    float x = (float) (rand() % (width - 2 * Constants::Yaal::MAX_FIELD_OF_VIEW)) + Constants::Yaal::MAX_FIELD_OF_VIEW;
    float y = (float) (rand() % (height - 2 * Constants::Yaal::MAX_FIELD_OF_VIEW)) + Constants::Yaal::MAX_FIELD_OF_VIEW;
    Vec2 pos = {x, y};
    return Plant(std::move(pos), std::move(body));
}
