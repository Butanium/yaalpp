#include "separablefilter.hpp"
#include "../cuda/diffusion_gpu.h"
#include <cmath>
#include <utility>

Tensor<float, 1> GaussianFilter(int size, float sigma) {
    Tensor<float, 1> filter(size);
    int center = size / 2;
    if (sigma == 0) {
        filter.setZero();
        filter(center) = 1;
        return filter;
    } else {
        // float norm = 1 / (sqrt(2 * (float) M_PI) * sigma);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float value = (float) exp(-pow((float) (i - center) / sigma, 2) / 2);
            filter(i) = value;
            sum += value;
        }
        for (int i = 0; i < size; i++) {
            filter(i) /= sum;
        }
        return filter;
    }
}

SeparableFilter::SeparableFilter(int filter_size, int nb_channels) :
        SeparableFilter(filter_size, nb_channels, BORDER_CONDITION_CLAMP, false, std::vector<float>(nb_channels, 1)) {
}

SeparableFilter::SeparableFilter(int filter_size, int nb_channels, bool skip_color_channels, std::vector<float> sigma) :
        SeparableFilter(filter_size, nb_channels, BORDER_CONDITION_CLAMP, skip_color_channels, std::move(sigma)) {
}

SeparableFilter::SeparableFilter(int filter_size, int nb_channels, int border_condition, bool skip_color_channels, std::vector<float> sigma) :
        filter_size(filter_size), nb_channels(nb_channels), border_condition(border_condition),
        skip_color_channels(skip_color_channels),
        use_cuda(false) {
    // the three first channels are for color, and are therefore not subject to pheromone diffusion.
    row_filters = Tensor<float, 2>(nb_channels, filter_size);
    col_filters = Tensor<float, 2>(nb_channels, filter_size);

    int start = skip_color_channels ? 3 : 0;
    for (int i = start; i < nb_channels; i++) {
        row_filters.chip(i, 0) = GaussianFilter(filter_size, sigma[i]);
        col_filters.chip(i, 0) = GaussianFilter(filter_size, sigma[i]);
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

void SeparableFilter::initialize_buffer(int start_c, int start_i, int start_j, const Tensor<float, 3> &input,
                                        Tensor<float, 1> &buffer, bool row_or_col, bool half_window) const {
    int buffer_size = buffer.dimension(0);
    int buffer_offset = half_window ? 0 : buffer_size / 2;
    int y = start_i;
    int x = start_j;
    for (int k = 0; k < buffer_size; k++) {
        if (row_or_col) {
            x = get_border_index(x - buffer_size + buffer_offset + k, 0, input.dimension(1));
        } else {
            y = get_border_index(y - buffer_size + buffer_offset + k, 0, input.dimension(0));
        }
        if (x == -1 || y == -1) {
            buffer(k) = 0;
        } else {
            buffer(k) = input(y, x, start_c);
        }
    }
}

void SeparableFilter::apply(const Tensor<float, 3> &input, Tensor<float, 3> &output) {
    apply(input, output, Offset(0, 0, 0, 0));
}

void SeparableFilter::apply(
        const Tensor<float, 3> &input,
        Tensor<float, 3> &output,
        Offset offset) {
  if (use_cuda) {
    cudaApply(input, output, offset);
    return;
  }

    // to avoid confusion :
    // indexing : i j c or y x c or height width channels
    // rows are at i fixed, columns at j fixed

    int height = input.dimension(0);
    int width = input.dimension(1);
    int channels = input.dimension(2);
    int half_filter_size = filter_size / 2;

    int start_c = skip_color_channels ? 3 : 0;

    Tensor<float, 3> temp = input;
    for (int c = start_c; c < channels; c++) {
        // 1-D convolution with the row filters for channel c
#pragma omp parallel for default(none) shared(temp, output, row_filters, col_filters, half_filter_size, width, height, c, offset) schedule(static)
        for (int i = offset.top; i < height - offset.bottom; i++) {
            // process row i of channel c
            for (int j = offset.left; j < width - offset.right; j++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int x = j - half_filter_size + k;
                    x = get_border_index(x, 0, width);
                    if (x == -1) {
                        continue;
                    }
                    sum += temp(i, x, c) * row_filters(c, k);
                }
                output(i, j, c) = sum;
            }
        }
        temp.chip(c, 2) = output.chip(c, 2);

        // 1-D convolution with the column filters for channel c
#pragma omp parallel for default(none) shared(temp, output, row_filters, col_filters, half_filter_size, width, height, c, offset) schedule(static)
        for (int j = offset.left; j < width - offset.right; j++) {
            // process column j of channel c
            for (int i = offset.top; i < height - offset.bottom; i++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int y = i - half_filter_size + k;
                    y = get_border_index(y, 0, height);
                    if (y == -1) {
                        continue;
                    }
                    sum += temp(y, j, c) * col_filters(c, k);
                }
                output(i, j, c) = sum;
            }
        }
    }
}


void SeparableFilter::apply_inplace(Tensor<float, 3> &input) {
    apply_inplace(input, Offset(0, 0, 0, 0));
}

void SeparableFilter::apply_inplace(Tensor<float, 3> &input, Offset offset) {
    if (use_cuda) {
        cudaApply(input, input, offset);
        return;
    }

    int height = input.dimension(0);
    int width = input.dimension(1);
    int channels = input.dimension(2);
    int half_filter_size = filter_size / 2;

    int start_c = skip_color_channels ? 3 : 0;

    for (int c = start_c; c < channels; c++) {
#pragma omp parallel for default(none) shared(input, row_filters, col_filters, half_filter_size, width, height, c, filter_size, offset) schedule(static)
        // 1-D convolution with the row filters for channel c
        for (int i = offset.top; i < height - offset.bottom; i++) {
            // initialise the buffer with the first values of the row, depending on the border condition
            Tensor<float, 1> buffer(filter_size);
            initialize_buffer(c, i, 0, input, buffer, true, true);

            // process row i of channel c
            for (int j = offset.left; j < width - offset.right; j++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int x = j - half_filter_size + k;
                    x = get_border_index(x, 0, width);
                    if (x == -1) {
                        continue;
                    }
                    if (k < half_filter_size) {
                        sum += buffer(x % half_filter_size) * row_filters(c, k);
                    } else {
                        sum += input(i, x, c) * row_filters(c, k);
                    }
                }
                buffer(j % half_filter_size) = input(i, j, c);
                input(i, j, c) = sum;
            }
        }
#pragma omp parallel for default(none) shared(input, row_filters, col_filters, half_filter_size, width, height, c, filter_size, offset) schedule(static)
        // 1-D convolution with the column filters for channel c
        for (int j = offset.left; j < width - offset.right; j++) {
            // initialise the buffer with the first values of the column
            Tensor<float, 1> buffer(half_filter_size);
            initialize_buffer(c, 0, j, input, buffer, false, true);

            // process column i of channel c
            for (int i = offset.top; i < height - offset.bottom; i++) {
                float sum = 0;
                for (int k = 0; k < filter_size; k++) {
                    int y = i - half_filter_size + k;
                    y = get_border_index(y, 0, height);
                    if (y == -1) {
                        continue;
                    }
                    if (k < half_filter_size) {
                        sum += buffer(y % half_filter_size) * col_filters(c, k);
                    } else {
                        sum += input(y, j, c) * col_filters(c, k);
                    }
                }
                buffer(i % half_filter_size) = input(i, j, c);
                input(i, j, c) = sum;
            }
        }
    }
}

void SeparableFilter::cudaApply(const Tensor<float, 3> &input, Tensor<float, 3> &output) {
    cudaApply(input, output, Offset(0, 0, 0, 0));
}

void SeparableFilter::cudaApply(const Tensor<float, 3> &input, Tensor<float, 3> &output, Offset offset) {
    int height = input.dimension(0);
    int width = input.dimension(1);
    int channels = input.dimension(2);
    int start_c = skip_color_channels ? 3 : 0;

    cudaFilterApply(input.data(), output.data(), width, height, channels, start_c, offset, filter_size, row_filters.data(), col_filters.data());
}
