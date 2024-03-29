/*
 * We will only deal with Gaussian filters since we will not use any other one.
 * A separable filter is one that can be expressed as the product of two vectors.
 * We can thus apply two 1D filters instead of a 2D filter, for a speedup of O(n) where n is the size of the filter.
 * */

#pragma once

#include "../utils/utils.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// Border Conditions :

#define BORDER_CONDITION_CLAMP 0
#define BORDER_CONDITION_WRAP 1
#define BORDER_CONDITION_REFLECT 2
#define BORDER_CONDITION_ZERO 3

using Eigen::Tensor;

Tensor<float, 1> GaussianFilter(int size, float sigma);

class SeparableFilter {
public:
    std::vector<float> sigma;

    Tensor<float, 2> row_filters;
    Tensor<float, 2> col_filters;

    Tensor<float, 2> row_filters_transpose;
    Tensor<float, 2> col_filters_transpose;

    int filter_size;
    int nb_channels;
    int border_condition;
    bool skip_color_channels;

public:
    bool use_cuda;

    SeparableFilter(int filter_size, int nb_channels);

    SeparableFilter(int filter_size, int nb_channels, int border_condition, bool skip_color_channels, std::vector<float> &&sigma_);

    SeparableFilter(int filter_size, int nb_channels, bool skip_color_channels, std::vector<float> &&sigma);

    ~SeparableFilter();

private:
    [[nodiscard]] int get_border_index(int index, int min, int max) const;

    void initialize_buffer(int start_c, int start_i, int start_j, const Tensor<float, 3>& input, Tensor<float, 1>& buffer, bool row_or_col, bool half_window) const;

public:
    // Apply the filter to the input tensor and store the result in the output tensor
    void apply(const Tensor<float, 3>& input, Tensor<float, 3>& output);

    void apply(const Tensor<float, 3> &input, Tensor<float, 3> &output, Offset offset);

    // Apply the filter to the input tensor and store the result in the output tensor using CUDA
    void cudaApply(const Tensor<float, 3>& input, Tensor<float, 3>& output);

    void cudaApply(const Tensor<float, 3>& input, Tensor<float, 3>& output, Offset offset);

    // Apply the filter inplace to the input tensor
    void apply_inplace(Tensor<float, 3>& input);

    void apply_inplace(Tensor<float, 3>& input, Offset offset);
};
