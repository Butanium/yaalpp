//
// Created by clementd on 31/01/24.
//

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#ifndef YAALPP_CREATURE_H
#define YAALPP_CREATURE_H

using Eigen::Tensor;
using Vec2 = Eigen::Vector2f;


enum struct YaalAction {
    Attack,
    Reproduce,
    Nop,
};

enum struct ActionSampling {
//    Softmax,
    Argmax,
//    Sample,
};

struct YaalDecision {
    YaalAction action;
    Vec2 direction;
    float speed_factor;
};

class YaalMLP {
public:
    Tensor<float, 1> direction_weights;
    Tensor<float, 1> speed_weights;
    float speed_bias;
    Tensor<float, 2> action_weights;
    Tensor<float, 1> decision_bias;
    ActionSampling decision_sampling;

    [[nodiscard]] Vec2 get_direction(const Tensor<float, 3> &input_view) const;

    YaalAction get_action(const Tensor<float, 3> &input_view) const;

    float get_speed_factor(const Tensor<float, 3> &input_view) const;

    YaalDecision evaluate(const Tensor<float, 3> &input_view) const;
};

struct YaalGenome {
    YaalMLP brain;
    float max_speed;
    int field_of_view;
    float max_size;
    float init_size;
};

struct YaalState {
    double health;
    double max_health;
    double energy;
    double max_energy;
    double age;
};


class Yaal {
public:
    YaalState internal_state;
    Vec2 position;
    Vec2 direction;
    float speed;
    YaalGenome genome;
    std::string sprite;

    void update();
};


#endif //YAALPP_CREATURE_H
