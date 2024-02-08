//
// Created by clementd on 31/01/24.
//

#include "Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include "../Constants.h"


using Eigen::Tensor;
using Eigen::array;
using Vec2 = Eigen::Vector2f;

Vec2 YaalMLP::get_direction(const Tensor<float, 3> &input_view) const {
    // direction_weights : (C)
    // input_view : (2F+1, 2F+1, C)
    // direction : (1,1)
    // Matrix product between each "pixel" of the view and the weights
    // Result is a (2F+1, 2F+1) weight map
    // Then compute the average direction weighted by the weight map
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(2, 0)};
    Tensor<float, 3> weight_map = input_view.contract(direction_weights, product_dims)
            .reshape(array<Eigen::Index, 3>{input_view.dimension(0), input_view.dimension(1), 1})
            .broadcast(array<Eigen::Index, 3>{1, 1, 2});
    // Create D: (2F+1, 2F+1, 2). D_ij is the direction from the F,F pixel to the i,j pixel
    // Init with the same height and width as the input view
    Tensor<float, 3> directions(input_view.dimension(0), input_view.dimension(1), 2);
    // Fill D with the direction vectors
    directions.setZero();
    for (int d = 0; d < 2; d++) {
        auto dim = input_view.dimension(d);
        float center;
        if (dim % 2 == 0) {
            center = (float) dim / 2.f - 0.5f;
        } else {
            center = (float) (dim / 2);
        }
        for (int i = 0; i < input_view.dimension(d); i++) {
            float cst;
            if (d == 0) {
                cst = (float) (i - center);
            } else {
                cst = (float) (center - i);
            }
            directions.chip(d, 2).chip(i, 1 - d).setConstant(cst);
        }
    }
    // Divide each direction vector by its norm
    Tensor<float, 3> d_norms = directions.square().sum(array<Eigen::Index, 1>{2}).sqrt().reshape(
            array<Eigen::Index, 3>{input_view.dimension(0), input_view.dimension(1), 1}).broadcast(
            array<Eigen::Index, 3>{1, 1, 2});
    if (input_view.dimension(0) % 2 && input_view.dimension(1) % 2) {
        d_norms.chip(input_view.dimension(0) / 2, 0).chip(input_view.dimension(1) / 2, 0).setConstant(1);
    }
    directions *= weight_map / d_norms;
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

YaalDecision YaalMLP::evaluate(const Tensor<float, 3> &input_view) const {
    return YaalDecision{
            .action = YaalAction::Nop,
            .direction = get_direction(input_view),
            .speed_factor = 1.0f,
    };
}

void Yaal::update(const Tensor<float, 3> &input_view) {
    auto decision = genome.brain.evaluate(input_view);
    position += decision.direction * (genome.max_speed * decision.speed_factor) * Constants::DELTA_T;
}

void Yaal::bound_position(const Vec2 &min, const Vec2 &max) {
    position = position.cwiseMax(min).cwiseMin(max);
}

Yaal::Yaal(Vec2&& position, YaalGenome&& genome) : position(std::move(position)), genome(std::move(genome)) {}

Yaal::Yaal(const Vec2 &position, const YaalGenome &genome) : position(position), genome(genome) {}

// TODO : is it ok with openmp ?
thread_local std::mt19937 YaalGenome::generator = std::mt19937(std::random_device{}());
thread_local std::mt19937 Yaal::generator = std::mt19937(std::random_device{}());

YaalGenome YaalGenome::random(int num_channels) {
    auto speed_rng = std::uniform_real_distribution<float>(Constants::Yaal::MIN_SPEED, Constants::Yaal::MAX_SPEED);
    auto fov_rng = std::uniform_int_distribution<int>(Constants::Yaal::MIN_FIELD_OF_VIEW,
                                                      Constants::Yaal::MAX_FIELD_OF_VIEW);
    auto size_rng = std::uniform_int_distribution<int>(Constants::Yaal::MIN_SIZE, Constants::Yaal::MAX_SIZE);
    return {
            .brain = YaalMLP{
                    .direction_weights = Tensor<float, 1>(num_channels).setRandom(),
            },
            .max_speed = speed_rng(generator),
            .field_of_view = fov_rng(generator),
            .size = size_rng(generator),
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
