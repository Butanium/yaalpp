//
// Created by clementd on 31/01/24.
//

#include "Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <iostream>
#include "../Constants.h"


using Eigen::Tensor;
using Eigen::array;
using Vec2 = Eigen::Vector2f;

Tensor<float, 3> direction_matrix(int height, int width) {
    // Compute the direction matrix s.t. D_ij is the normalized direction from the center to the i,j pixel
    int dims[] = {height, width};
    Tensor<float, 3> directions(height, width, 2);
    directions.setZero();
    for (int d = 0; d < 2; d++) {
        float center = (float) dims[d] / 2.f - 0.5f;
        for (int i = 0; i < dims[d]; i++) {
            directions.chip(d, 2).chip(i, 1 - d).setConstant((float) (i - center));
        }
    }
    Tensor<float, 3> d_norms = directions.square().sum(array<Eigen::Index, 1>{2}).sqrt().reshape(
            array<Eigen::Index, 3>{height, width, 1}).broadcast(
            array<Eigen::Index, 3>{1, 1, 2});
    if (height % 2 && width % 2) {
        d_norms.chip(height / 2, 0).chip(width / 2, 0).setConstant(1);
    }
    return directions / d_norms;
}
// Eigen::Tensor<float, 3, 0, long>
//template<typename T> // Eigen::TensorSlicingOp<std::array<long, 3ul> const, std::array<long, 3ul> const, Eigen::Tensor<float, 3, 0, long>>
Vec2 YaalMLP::get_direction(Eigen::TensorSlicingOp<std::array<long, 3ul> const, std::array<long, 3ul> const, Eigen::Tensor<float, 3, 0, long>>  input_view, int height, int width) const {
    // direction_weights : (C)
    // input_view : (2F+1, 2F+1, C)
    // direction : (1,1)
    // Matrix product between each "pixel" of the view and the weights
    // Result is a (2F+1, 2F+1) weight map
    // Then compute the average direction weighted by the weight map
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(2, 0)};
    Tensor<float, 3> weight_map = input_view.contract(direction_weights, product_dims)
            .reshape(array<Eigen::Index, 3>{height, width, 1})
            .broadcast(array<Eigen::Index, 3>{1, 1, 2});
    // Create D: (2F+1, 2F+1, 2). D_ij is the direction from the F,F pixel to the i,j pixel
    // Init with the same height and width as the input view
    auto directions = direction_matrix(height, width);
    directions *= weight_map;
    Tensor<float, 0> x = directions.chip(0, 2).mean();
    Tensor<float, 0> y = directions.chip(1, 2).mean();
    Vec2 direction = {x(0), y(0)};
    auto norm = direction.norm();
    if (norm < Constants::EPSILON) {
        return Vec2::Zero();
    }
    direction.normalize();
    return direction;
}

//template<typename T>
YaalDecision YaalMLP::evaluate(Eigen::TensorSlicingOp<std::array<long, 3ul> const, std::array<long, 3ul> const, Eigen::Tensor<float, 3, 0, long>> input_view, int height, int width) const {
    return YaalDecision{
            .direction = get_direction(input_view, height, width),
            .speed_factor = 1.0f,
    };
}

template<typename T>
void Yaal::update(T  input_view) {
    auto decision = genome.brain.evaluate(input_view, genome.field_of_view * 2 + genome.size,
                                          genome.field_of_view * 2 + genome.size);
    position += decision.direction * (genome.max_speed * decision.speed_factor) * Constants::DELTA_T;
}

void Yaal::bound_position(const Vec2 &min, const Vec2 &max) {
    position = position.cwiseMax(min).cwiseMin(max);
}

Yaal::Yaal(Vec2 &&position, YaalGenome &&genome) : position(std::move(position)), genome(std::move(genome)) {}

Yaal::Yaal(const Vec2 &position, const YaalGenome &genome) : position(position), genome(genome) {}

// TODO : is it ok with openmp ?
thread_local std::mt19937 YaalGenome::generator = std::mt19937(std::random_device{}());
thread_local std::mt19937 Yaal::generator = std::mt19937(std::random_device{}());


Tensor<float, 3> YaalGenome::generate_body(int size, std::array<float, 3> color) {
    Tensor<float, 3> body(size, size, 4);
    for (int c = 0; c < 3; c++)
        body.chip(c, 2).setConstant(color[c]);
    body.chip(3, 2).setConstant(1);
    // Apply circle mask
    float center = (float) size / 2.f - 0.5f;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = (float) i - center;
            float y = (float) j - center;
            auto slice = body.chip(i, 0).chip(j, 0);
            if (x * x + y * y > size * size / 4.f) {
                slice.setZero();
            } else {
                // Interpolate between *= 0.5 and *= 1
//                slice *= 0.5f + 0.5f * (1 - std::sqrt(x * x + y * y) / (size / 2.f));
            }
        }
    }
    return body;
}


YaalGenome YaalGenome::random(int num_channels) {
    auto speed_rng = std::uniform_real_distribution<float>(Constants::Yaal::MIN_SPEED, Constants::Yaal::MAX_SPEED);
    auto fov_rng = std::uniform_int_distribution<int>(Constants::Yaal::MIN_FIELD_OF_VIEW,
                                                      Constants::Yaal::MAX_FIELD_OF_VIEW);
    auto size_rng = std::uniform_int_distribution<int>(Constants::Yaal::MIN_SIZE, Constants::Yaal::MAX_SIZE);
    auto color_rng = std::uniform_real_distribution<float>(0, 1);
    int size = size_rng(generator);
    std::array<float, 3> color = {color_rng(generator), color_rng(generator), color_rng(generator)};
    auto body = generate_body(size, color);
    return {
            .brain = YaalMLP{
                    .direction_weights = Tensor<float, 1>(num_channels).setRandom(),
            },
            .max_speed = speed_rng(generator),
            .field_of_view = fov_rng(generator),
            .size = size,
            .body = body,
            .color = color
    };
}

Yaal Yaal::random(int num_channels, const Vec2 &position) {
    return {position, YaalGenome::random(num_channels)};
}

void Yaal::setRandomPosition(const Vec2 &min, const Vec2 &max) {
    std::uniform_real_distribution<float> x_rng(min.x(), max.x());
    std::uniform_real_distribution<float> y_rng(min.y(), max.y());
    position = {x_rng(generator), y_rng(generator)};
}

Yaal Yaal::random(int num_channels) {
    return random(num_channels, Vec2::Zero());
}
