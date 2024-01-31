//
// Created by clementd on 31/01/24.
//

#include "Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

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
    Tensor<float, 3> D(input_view.dimension(0), input_view.dimension(1), 2);
    // Fill D with the direction vectors
    D.setZero();
    for (int d = 0; d < 2; d++) {
        auto center = input_view.dimension(d) / 2;
        for (int i = 0; i < input_view.dimension(d); i++) {
            D.chip(d, 2).chip(i, d).setConstant((float) (i - center));
        }
    }
    // Divide each direction vector by its norm
    auto D_norm = D.square().sum(array<Eigen::Index, 1>{2}).sqrt().reshape(
            array<Eigen::Index, 3>{input_view.dimension(0), input_view.dimension(1), 1}).broadcast(
            array<Eigen::Index, 3>{1, 1, 2});
    D *= weight_map / D_norm;
    Tensor<float, 0> x = D.chip(0, 2).mean();
    Tensor<float, 0> y = D.chip(1, 2).mean();
    return Vec2(x(0), y(0));
}