#ifndef DIFFUSION_GPU_H
#define DIFFUSION_GPU_H

#include "../utils/utils.h"

// Low level to apply filter through CUDA
void cudaFilterApply(const float *input, float *output, int width, int height, int channels, int start_channels, Offset offset, int filter_size, float *row_filter, float *col_filter);

#endif // !DIFFUSION_GPU_H
