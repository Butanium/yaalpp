//
// Created by clementd on 31/01/24.
//

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Constants.h"

#ifndef YAALPP_CREATURE_H
#define YAALPP_CREATURE_H

using Eigen::Tensor;
using Vec2 = Eigen::Vector2f;
using Eigen::array;

Tensor<float, 3> direction_matrix(int height, int width);

/// The possible actions for a Yaal
//enum struct YaalAction {
//    Attack,
//    Reproduce,
//    Nop,
//};


//enum struct ActionSampling {
////    Softmax,
//    Argmax,
////    Sample,
//};

/// The decision of a Yaal
struct YaalDecision {
//    YaalAction action;
    Vec2 direction;
    float speed_factor;
};

/**
 * A Multi-Layer Perceptron for Yaal's brain
 */
class YaalMLP {
    [[nodiscard]] Vec2 get_direction(auto &input_view, int height, int width) const {
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
        assert (!direction.hasNaN());
        return direction;
    }


public:
    Tensor<float, 1> direction_weights;
//    Tensor<float, 1> speed_weights;
//    float speed_bias;
//    Tensor<float, 2> action_weights;
//    Tensor<float, 1> decision_bias;
//    ActionSampling decision_sampling;


//    YaalAction get_action(const Tensor<float, 3> &input_view) const;

//    float get_speed_factor(const Tensor<float, 3> &input_view) const;
    /**
     * Evaluate the MLP on the given input view
     * @param input_view The input view
     * @return The Yaal's decision
     */
    [[nodiscard]] YaalDecision evaluate(auto &input_view, int height, int width) const {
        return YaalDecision{
                .direction = get_direction(input_view, height, width),
                .speed_factor = 1.0f,
        };
    }
};

/// The genome of a Yaal. Contains the brain and other fixed parameters
class YaalGenome {
public:
    YaalMLP brain;
    float max_speed;
    int field_of_view;
    int size;
    Tensor<float, 3> body;
    std::vector<float> signature;

    static Tensor<float, 3> generate_body(int size, const std::vector<float> &signature);

    static YaalGenome random(int num_channels);
//    float max_size;
//    float init_size;
    static thread_local std::mt19937 generator;
};

/** The state of a Yaal.
 * Contains the health, energy, age, etc. Those are the parameters that can change during the Yaal's life
 */
//struct YaalState {
//    double health;
//    double max_health;
//    double energy;
//    double max_energy;
//    double age;
//};

/**
 * A Yaal
 * This is the class of the creatures that will evolve in the simulation
 */
class Yaal {
public:
    /**
     * Construct a Yaal
     * @param position The initial position
     * @param genome The genome
     */
    Yaal(Vec2 &&position, YaalGenome &&genome);

    Yaal(const Vec2 &position, const YaalGenome &genome);

    static thread_local std::mt19937 generator;

/**
     * Generate a random Yaal
     */
    static Yaal random(int num_channels, const Vec2 &position);

    static Yaal random(int num_channels);


//    YaalState internal_state;
    Vec2 position;
    Vec2 direction;
    YaalGenome genome;

    /**
     * Update the Yaal's state position, direction, speed, etc.
     * @param input_view What the Yaal sees
     */
    void update(auto &input_view) {
        auto decision = genome.brain.evaluate(input_view, genome.field_of_view * 2 + genome.size,
                                              genome.field_of_view * 2 + genome.size);
        position += decision.direction * (genome.max_speed * decision.speed_factor) * Constants::DELTA_T;
    }

    void setRandomPosition(const Vec2 &min, const Vec2 &max);

    /**
     * Bound the Yaal's position between min and max
     * @param min, max The bounds
     */
    void bound_position(const Vec2 &min, const Vec2 &max);

    /**
     * Return the position of the top left corner of the Yaal's bodyf
     */
    Vec2 top_left_position() const;
};


#endif //YAALPP_CREATURE_H
