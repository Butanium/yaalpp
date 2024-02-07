//
// Created by clementd on 31/01/24.
//

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#ifndef YAALPP_CREATURE_H
#define YAALPP_CREATURE_H

using Eigen::Tensor;
using Vec2 = Eigen::Vector2f;

/// The possible actions for a Yaal
enum struct YaalAction {
    Attack,
    Reproduce,
    Nop,
};


//enum struct ActionSampling {
////    Softmax,
//    Argmax,
////    Sample,
//};

/// The decision of a Yaal
struct YaalDecision {
    YaalAction action;
    Vec2 direction;
    float speed_factor;
};

/**
 * A Multi-Layer Perceptron for Yaal's brain
 */
class YaalMLP {
    [[nodiscard]] Vec2 get_direction(const Tensor<float, 3> &input_view) const;

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
    [[nodiscard]] YaalDecision evaluate(const Tensor<float, 3> &input_view) const;
};

/// The genome of a Yaal. Contains the brain and other fixed parameters
class YaalGenome {
    static thread_local std::mt19937 generator;
public:
    YaalMLP brain;
    float max_speed;
    int field_of_view;
    int size;
    static YaalGenome random(int num_channels);
//    float max_size;
//    float init_size;
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
    Yaal(Vec2 position, YaalGenome genome);
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
    void update(const Tensor<float, 3> &input_view);

    void setRandomPosition(const Vec2 &min, const Vec2 &max);

    /**
     * Bound the Yaal's position between min and max
     * @param min, max The bounds
     */
    void bound_position(const Vec2 &min, const Vec2 &max);
};


#endif //YAALPP_CREATURE_H
