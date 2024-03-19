#include "diffusion_gpu.h"
#include <cuda.h>
#include "../utils/utils.h"
#include <stdio.h>

__global__ void filterApplyColKernel(float *input, float *output, int width, int height, int channels, Offset offset, int filter_size, float *col_filter) {
  // x = width, y = height, z = channels
  // ORDER : height -> width -> channels
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int ic = blockIdx.z * blockDim.z + threadIdx.z;

  for (int x = ix; x < width - offset.right; x += gridDim.x * blockDim.x) {
    for (int y = iy; y < height - offset.bottom; y += gridDim.y * blockDim.y) {
      for(int c = ic; c < channels; c += gridDim.z * blockDim.z) {
        if (offset.left <= x && x < width - offset.right && offset.top <= y && y < height - offset.bottom && c < channels) { // Additional check useful ?
          int index = ( c * width + x ) * height + y;

          int y_start = (c * width + x) * height;
          int y_end = y_start + height;

          output[index] = 0.0f;
          for (int i = 0; i < filter_size; i++) {
            int y_index = index - filter_size / 2 + i;
            if (y_index < y_start) y_index = y_start;
            if (y_index >= y_end) y_index = y_end - 1;

            output[index] += input[y_index] * col_filter[c * filter_size + i];
          }
        }
      }
    }
  }
}

__global__ void filterApplyRowKernel(float *input, float *output, int width, int height, int channels, Offset offset, int filter_size, float *row_filter) {
  // x = width, y = height, z = channels
  // ORDER : height -> width -> channels
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int ic = blockIdx.z * blockDim.z + threadIdx.z;

  for (int x = ix; x < width - offset.right; x += gridDim.x * blockDim.x) {
    for (int y = iy; y < height - offset.bottom; y += gridDim.y * blockDim.y) {
      for(int c = ic; c < channels; c += gridDim.z * blockDim.z) {
        if (offset.left <= x && x < width - offset.right && offset.top <= y && y < height - offset.bottom && c < channels) {
          int index = ( c * width + x ) * height + y;

          int x_start = (c * width + 0) * height + y;
          int x_end = x_start + width * height;

          output[index] = 0.0f;
          for (int i = 0; i < filter_size; i++) {
            int x_index = index + ( i - filter_size / 2 ) * height;
            if (x_index < x_start) x_index = x_start;
            if (x_index >= x_end) x_index = x_end - 1;

            output[index] += input[x_index] * row_filter[c * filter_size + i];
          }
        }
      }
    }
  }
}


void cudaFilterApply(const float *input, float *output, int width, int height, int channels, int start_channels, Offset offset, int filter_size, float *row_filter, float *col_filter) {
  int num_channels = channels - start_channels;

  float *d_input, *d_tmp, *d_row_filter, *d_col_filter;
  int size = width * height * num_channels * sizeof(float);
  int filter_size_bytes = filter_size * num_channels * sizeof(float);

  // Allocate memory
  cudaMalloc((void **)&d_input, size);
  cudaMalloc((void **)&d_tmp, size);
  cudaMalloc((void **)&d_row_filter, filter_size_bytes);
  cudaMalloc((void **)&d_col_filter, filter_size_bytes);

  // Copy data to device
  cudaMemcpy(d_input, input + start_channels * width * height, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_filter, row_filter + start_channels * filter_size, filter_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_filter, col_filter + start_channels * filter_size, filter_size_bytes, cudaMemcpyHostToDevice);

  // run row & col kernels
  filterApplyRowKernel<<<dim3(16, 16, num_channels), dim3(16, 16, 1)>>>(d_input, d_tmp, width, height, num_channels, offset, filter_size, d_row_filter);
  filterApplyColKernel<<<dim3(16, 16, num_channels), dim3(16, 16, 1)>>>(d_tmp, d_input, width, height, num_channels, offset, filter_size, d_col_filter);

  // kernel successful ?
  // cudaError_t err = cudaGetLastError();
  // printf("Error : %s\n", cudaGetErrorString(err));

  // Copy data back to host
  cudaMemcpy(output + start_channels * width * height, d_input, size, cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_input);
  cudaFree(d_tmp);
  cudaFree(d_row_filter);
  cudaFree(d_col_filter);
}
