//
// Created by clementd on 31/01/24.
//
#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Constants.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>


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

struct SerializedYaalMLP {
    std::vector<float> direction_weights;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & direction_weights;
    }
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

    SerializedYaalMLP to_serialized() const {
        SerializedYaalMLP serialized;
        serialized.direction_weights = std::vector<float>(direction_weights.data(),
                                                          direction_weights.data() + direction_weights.size());
        return serialized;
    }

    static YaalMLP from_serialized(const SerializedYaalMLP &serialized) {
        Tensor<float, 1> dir_weights = Eigen::TensorMap<Tensor<float, 1>>(
                const_cast<float *>(serialized.direction_weights.data()), serialized.direction_weights.size());
        return {.direction_weights = dir_weights,};
    }
};


struct SerializedYaalGenome {
    SerializedYaalMLP brain;
    float max_speed;
    int field_of_view;
    int size;
    std::vector<float> signature;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & brain;
        ar & max_speed;
        ar & field_of_view;
        ar & size;
        ar & signature;
    }
};

/// The genome of a Yaal. Contains the brain and other fixed parameters
class YaalGenome {

public:
    YaalMLP brain;
    float max_speed;
    int field_of_view;
    int size;
    std::vector<float> signature;

    Tensor<float, 3> generate_body();

    static YaalGenome random(int num_channels);

//    float max_size;
//    float init_size;
    static std::mt19937 generator;

    SerializedYaalGenome to_serialized() const {
        SerializedYaalGenome serialized;
        serialized.brain = brain.to_serialized();
        serialized.max_speed = max_speed;
        serialized.field_of_view = field_of_view;
        serialized.size = size;
        serialized.signature = signature;
        return serialized;
    }

    static YaalGenome from_serialized(const SerializedYaalGenome &serialized) {
        YaalGenome yaalGenome;
        yaalGenome.brain = YaalMLP::from_serialized(serialized.brain);
        yaalGenome.max_speed = serialized.max_speed;
        yaalGenome.field_of_view = serialized.field_of_view;
        yaalGenome.size = serialized.size;
        yaalGenome.signature = serialized.signature;
        return yaalGenome;
    }
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

struct SerializedYaal {
    float position_x;
    float position_y;
    SerializedYaalGenome genome;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & position_x;
        ar & position_y;
        ar & genome;
    }
};

/**
 * A Yaal
 * This is the class of the creatures that will evolve in the simulation
 */
class Yaal {

public:
    //    YaalState internal_state;
    Vec2 position;
    Vec2 direction;
    YaalGenome genome;
    Tensor<float, 3> body;

    /**
     * Construct a Yaal
     * @param position The initial position
     * @param genome The genome
     */
    Yaal(Vec2 &&position, YaalGenome &&genome, Tensor<float, 3> &&body);

    Yaal(const Vec2 &position, const YaalGenome &genome, const Tensor<float, 3> &body);

    static std::mt19937 generator;

    /// Generate a random Yaal
    static Yaal random(int num_channels, const Vec2 &position);

    static Yaal random(int num_channels);


    /**
     * Update the Yaal's state position, direction, speed, etc.
     * @param input_view What the Yaal sees
     */
    void update(auto &input_view) {
        auto decision = genome.brain.evaluate(input_view, genome.field_of_view * 2 + genome.size,
                                              genome.field_of_view * 2 + genome.size);
        position += decision.direction * (genome.max_speed * decision.speed_factor) * Constants::DELTA_T;
    }

    void set_random_position(const Vec2 &min, const Vec2 &max);

    /**
     * Bound the Yaal's position between min and max
     * @param min, max The bounds
     */
    void bound_position(const Vec2 &min, const Vec2 &max);

    /**
     * Return the position of the top left corner of the Yaal's bodyf
     */
    Vec2 top_left_position() const;

    SerializedYaal to_serialized() const {
        SerializedYaal serialized;
        serialized.position_x = position.x();
        serialized.position_y = position.y();
        serialized.genome = genome.to_serialized();
        return serialized;
    }

    static Yaal from_serialized(const SerializedYaal &serialized) {
        YaalGenome genome = YaalGenome::from_serialized(serialized.genome);
        Yaal yaal(Vec2(serialized.position_x, serialized.position_y),
                  std::move(genome),
                  std::move(genome.generate_body()));
        return yaal;
    }
};
