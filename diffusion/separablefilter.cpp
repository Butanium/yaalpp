#include "separablefilter.hpp"
#include <cmath>
#include <omp.h>

Tensor<float, 1> GaussianFilter(int size, float sigma) {
    Tensor<float, 1> filter(size);
    int center = size / 2;
    float norm = 1 / (sqrt(2 * (float) M_PI) * sigma);
    for (int i = 0; i < size; i++)
    {
        filter(i) = (float) exp(-pow((float) (i - center) / sigma, 2) / 2) / norm;
    }
    return filter;
}

SeparableFilter::SeparableFilter(int filter_size, int nb_channels) :
SeparableFilter(filter_size, nb_channels, BORDER_CONDITION_CLAMP, false) {
}

SeparableFilter::SeparableFilter(int filter_size, int nb_channels, int border_condition, bool skip_color_channels) :
    filter_size(filter_size), nb_channels(nb_channels), border_condition(border_condition), skip_color_channels(skip_color_channels) {
    // the three first channels are for color, and are therefore not subject to pheromone diffusion.
    row_filters = Tensor<float, 2>(nb_channels, filter_size);
    col_filters = Tensor<float, 2>(nb_channels, filter_size);

    // TODO : different sigma for each channel
    int start = skip_color_channels ? 3 : 0;
    for (int i = start; i < nb_channels; i++) {
        row_filters.chip(i, 0) = GaussianFilter(filter_size, 1);
        col_filters.chip(i, 0) = GaussianFilter(filter_size, 1);
    }
}

SeparableFilter::~SeparableFilter() = default;

int SeparableFilter::get_border_index(int index, int min, int max) const {
    switch (border_condition) {
        case BORDER_CONDITION_CLAMP:
            if (index < min) {
                return min;
            } else if (index >= max) {
                return max - 1;
            }
            return index;
        case BORDER_CONDITION_WRAP:
            if (index < min) {
                return max + index;
            } else if (index >= max) {
                return index - max;
            }
            return index;
        case BORDER_CONDITION_REFLECT:
            if (index < min) {
                return min - index;
            } else if (index >= max) {
                return max - (index - max) - 1;
            }
            return index;
        case BORDER_CONDITION_ZERO:
            if (index < min || index >= max) {
                return -1;
            }
            return index;
        default:
            return -1;
    }
}

void SeparableFilter::initialize_buffer(int start_c, int start_i, int start_j, const Tensor<float, 3> &input, Tensor<float, 1> &buffer, bool row_or_col, bool half_window) {
    int buffer_size = buffer.dimension(0);
    int buffer_offset = half_window ? 0 : buffer_size / 2;
    for (int k = 0; k < buffer_size; k++) {
        int x = start_i;
        int y = start_j;
        if (row_or_col) {
            x = get_border_index(x - buffer_size + buffer_offset + k, 0, input.dimension(0));
        } else {
            y = get_border_index(y - buffer_size + buffer_offset + k, 0, input.dimension(1));
        }
        if (x == -1 || y == -1) {
            buffer(k) = 0;
        } else {
            buffer(k) = input(x, y, start_c);
        }
    }
}

void SeparableFilter::apply(const Tensor<float, 3> &input, Tensor<float, 3> &output) {
    // TODO : is the input size c h w or w h c ? RESOLVED : h w c (y x c)
    // TODO : offset (for the borders when using MPI). Just start and end the loops at +- border_size
    // TODO : is the order really significant ? With the "correct" order, the values should be most of the time much closer in the processor
    //        cache or even registers, so it should be much faster. (note : previous order was for i j c)
    // TODO : OpenMP parallel on row or col, not on channels. Try on channel, but parallelization should be much weaker.
    //        Try static and dynamic scheduling.

    int width = input.dimension(0);
    int height = input.dimension(1);
    int channels = input.dimension(2);
    int half_filter_size = filter_size / 2;

    int start_c = skip_color_channels ? 3 : 0;

    Tensor<float, 3> temp = input;
    for (int c = start_c; c < channels; c++) {
        // 1-D convolution with the row filters for channel c
#pragma omp parallel for default(none) shared(temp, output, row_filters, col_filters, half_filter_size, width, height, c) schedule(static)
        for (int j = 0; j < height; j++) {
            // process row j of channel c
            for (int i = 0; i < width; i++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int x = i - half_filter_size + k;
                    x = get_border_index(x, 0, width);
                    if (x == -1) {
                        continue;
                    }
                    sum += temp(x, j, c) * row_filters(c, k);
                }
                output(i, j, c) = sum;
            }
        }
        temp.chip(c, 2) = output.chip(c, 2);

        // 1-D convolution with the column filters for channel c
#pragma omp parallel for default(none) shared(temp, output, row_filters, col_filters, half_filter_size, width, height, c) schedule(static)
        for (int i = 0; i < width; i++) {
            // process column i of channel c
            for (int j = 0; j < height; j++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int y = j - half_filter_size + k;
                    y = get_border_index(y, 0, height);
                    if (y == -1) {
                        continue;
                    }
                    sum += temp(i, y, c) * col_filters(c, k);
                }
                output(i, j, c) = sum;
            }
        }
    }
}

void SeparableFilter::apply_inplace(Tensor<float, 3> &input) {
    // TODO's : same as apply
    // TODO : is it faster if we store the whole window filter_size in the buffer instead of only half, to avoid the if and only access to one array ?

    int width = input.dimension(0);
    int height = input.dimension(1);
    int channels = input.dimension(2);
    int half_filter_size = filter_size / 2;

    int start_c = skip_color_channels ? 3 : 0;

    // The buffer is used to cyclically store the erased values so that the next step can still use them
    Tensor<float, 1> buffer(half_filter_size);

    for (int c = start_c; c < channels; c++) {
#pragma omp parallel for default(none) shared(input, row_filters, col_filters, half_filter_size, width, height, c, filter_size) private(buffer) schedule(static)
        // 1-D convolution with the row filters for channel c
        for (int j = 0; j < height; j++) {
            // initialise the buffer with the first values of the row, depending on the border condition
            initialize_buffer(c, 0, j, input, buffer, true, true);

            // process row j of channel c
            for (int i = 0; i < width; i++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int x = i - half_filter_size + k;
                    x = get_border_index(x, 0, width);
                    if (x == -1) {
                        continue;
                    }
                    if (k < half_filter_size) {
                        sum += buffer(x % half_filter_size) * row_filters(c, k);
                    } else {
                        sum += input(x, j, c) * row_filters(c, k);
                    }
                }
                buffer(i % half_filter_size) = input(i, j, c);
                input(i, j, c) = sum;
            }
        }
#pragma omp parallel for default(none) shared(input, row_filters, col_filters, half_filter_size, width, height, c, filter_size) private(buffer) schedule(static)
        // 1-D convolution with the column filters for channel c
        for (int i = 0; i < width; i++) {
            // initialise the buffer with the first values of the column
            initialize_buffer(c, i, 0, input, buffer, false, true);

            // process column i of channel c
            for (int j = 0; j < height; j++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int y = j - half_filter_size + k;
                    y = get_border_index(y, 0, height);
                    if (y == -1) {
                        continue;
                    }
                    if (k < half_filter_size) {
                        sum += buffer(y % half_filter_size) * col_filters(c, k);
                    } else {
                        sum += input(i, y, c) * col_filters(c, k);
                    }
                }
                buffer(j % half_filter_size) = input(i, j, c);
                input(i, j, c) = sum;
            }
        }
    }
}