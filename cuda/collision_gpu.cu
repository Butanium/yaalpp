#include <iostream>
#include <cuda.h>

#define CUDA_CHECK_ERROR() { cudaError_t err = cudaGetLastError(); \
                              if (err != cudaSuccess) { \
                                  printf("CUDA error: %s\n", cudaGetErrorString(err)); \
                                  exit(-1); \
                              } \
                            }

// diff between v1 and v2 : order of storage. v1 : each yaal has its distances contiguously, v2 : each entity has its distances contiguously.
__global__ void collisionComputeDistancev1(const float *input_x, const float *input_y, float *output, int nb_yaals, int nb_entities) {
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start_j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        for (int j = start_j; j < nb_entities; j += blockDim.y * gridDim.y) {
            float dx = input_x[j] - input_x[i];
            float dy = input_y[j] - input_y[i];
            output[i * nb_entities + j] = dx * dx + dy * dy;
        }
    }
}

__global__ void collisionComputeDistancev2(const float *input_x, const float *input_y, float *output, int nb_yaals, int nb_entities) {
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start_j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        for (int j = start_j; j < nb_entities; j += blockDim.y * gridDim.y) {
            float dx = input_x[j] - input_x[i];
            float dy = input_y[j] - input_y[i];
            output[j * nb_yaals + i] = dx * dx + dy * dy;
        }
    }
}

// test to see if can speedup gather v1 by contiguous access (spoiler : no)
__global__ void transpose(float *input, int nb_yaals, int nb_entities) {
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start_j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        for (int j = start_j; j < nb_entities; j += blockDim.y * gridDim.y) {
            if (i < j) {
                float tmp = input[i * nb_entities + j];
                input[i * nb_entities + j] = input[j * nb_yaals + i];
                input[j * nb_yaals + i] = tmp;
            }
        }
    }
}

/*
CUDA kernel to gather the closest entity for each yaal.
*/
__global__ void collisionGatherClosestv1(const float *distances, float *output_dist, int *output_index, int nb_yaals, int nb_entities) {
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        float min_dist = -1;
        int min_index = -1;
        for (int j = 0; j < nb_entities; j++) {
            float dist = distances[i * nb_entities + j];
            if ((dist < min_dist || min_dist < 0) && dist > 0) {
                min_dist = dist;
                min_index = j;
            }
        }
        output_index[i] = min_index;
        output_dist[i] = min_dist;
    }
}

__global__ void collisionGatherClosestv2(const float *distances, float *output_dist, int *output_index, int nb_yaals, int nb_entities) {
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        float min_dist = -1;
        int min_index = -1;
        for (int j = 0; j < nb_entities; j++) {
            float dist = distances[j * nb_yaals + i];
            if ((dist < min_dist || min_dist < 0) && dist > 0) {
                min_dist = dist;
                min_index = j;
            }
        }
        output_index[i] = min_index;
        output_dist[i] = min_dist;
    }
}

// recompute dist matrix
__global__ void collisionGatherClosestv3(const float *input_x, const float *input_y, int *output_index, float *output_dist, int nb_yaals, int nb_entities) {
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        float i_x = input_x[i];
        float i_y = input_y[i];
        float min_dist = -1;
        int min_index = -1;
        for (int j = 0; j < nb_entities; j++) {
            float j_x = input_x[j];
            float j_y = input_y[j];
            float dist = (i_x - j_x) * (i_x - j_x) + (i_y - j_y) * (i_y - j_y);
            if ((dist < min_dist || min_dist < 0) && dist > 0) {
                min_dist = dist;
                min_index = j;
            }
        }
        output_index[i] = min_index;
        output_dist[i] = min_dist;
    }
}

/*
CUDA kernel to gather the closest entity for each yaal.
Instead of having one thread per yaal doing a for loop on the entities, have one thread per yaal per two entities that decides which one is the closest, and apply it recursively until we have only one entity left.
The same number of operations is done, but parallelism is increased.

We have to change the storage of the distances so that they remain contiguous in memory.
TODO : faster ?
*/
__global__ void collisionGatherClosestv4(const float *input_distances, const int* input_indices, float *output_distances, int *output_indices, int nb_yaals, int nb_entities, bool index_already_initialized) {
    // nb_entities is not the true number of entities, but the number of comparisons to do at this step.
    int start_i = blockIdx.x * blockDim.x + threadIdx.x;
    int start_j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = start_i; i < nb_yaals; i += blockDim.x * gridDim.x) {
        for (int j = start_j; j < nb_entities / 2 + nb_entities % 2; j += blockDim.y * gridDim.y) {
            if (j * 2 + 1 < nb_entities) {
                int index1, index2;
                if (index_already_initialized) {
                    index1 = input_indices[i * nb_entities + j * 2];
                    index2 = input_indices[i * nb_entities + j * 2 + 1];
                } else {
                    index1 = j * 2;
                    index2 = j * 2 + 1;
                }
                float dist1 = input_distances[i * nb_entities + j * 2];
                float dist2 = input_distances[i * nb_entities + j * 2 + 1];
                if ((dist1 < dist2 && dist1 > 0) || dist2 < 1e-12) {
                    output_indices[i] = index1;
                    output_distances[i] = dist1;
                } else {
                    output_indices[i] = index2;
                    output_distances[i] = dist2;
                }
            } else {
                if (index_already_initialized) {
                    output_indices[i] = input_indices[i * nb_entities + j * 2];
                    output_distances[i] = input_distances[i * nb_entities + j * 2];
                } else {
                    output_indices[i] = j * 2;
                    output_distances[i] = input_distances[i * nb_entities + j * 2];
                }
            }
        }
    }
}

void cudaCollisionApplyv1(const float *d_input_x, const float *d_input_y, float* d_dist, float *d_output_dist, int *d_output_index, int nb_yaals, int nb_entities) {
    // input size : nb_entities * 2
    // output size : nb_yaals * 2

    // tests, +- on x and y and report as plot.
    dim3 threadsPerBlock(2, 128);
    int grid_x = min((nb_yaals + threadsPerBlock.x - 1) / threadsPerBlock.x, 64);
    int grid_y = min((nb_entities + threadsPerBlock.y - 1) / threadsPerBlock.y, 64);
    dim3 numBlocks(grid_x, grid_y);
    collisionComputeDistancev1<<<numBlocks, threadsPerBlock>>>(d_input_x, d_input_y, d_dist, nb_yaals, nb_entities);
    CUDA_CHECK_ERROR();

    int block_size = 256;
    int grid_size = min((nb_yaals + block_size - 1) / block_size, 64);
    collisionGatherClosestv1<<<grid_size, block_size>>>(d_dist, d_output_dist, d_output_index, nb_yaals, nb_entities);
}

void cudaCollisionApplyv2(const float *d_input_x, const float *d_input_y, float* d_dist, float *d_output_dist, int *d_output_index, int nb_yaals, int nb_entities) {
    // input size : nb_entities * 2
    // output size : nb_yaals * 2

    // tests, +- on x and y and report as plot.
    dim3 threadsPerBlock(128, 2);
    int grid_x = min((nb_yaals + threadsPerBlock.x - 1) / threadsPerBlock.x, 64);
    int grid_y = min((nb_entities + threadsPerBlock.y - 1) / threadsPerBlock.y, 64);
    dim3 numBlocks(grid_x, grid_y);
    collisionComputeDistancev2<<<numBlocks, threadsPerBlock>>>(d_input_x, d_input_y, d_dist, nb_yaals, nb_entities);
    CUDA_CHECK_ERROR();

    int block_size = 256;
    int grid_size = min((nb_yaals + block_size - 1) / block_size, 64);
    collisionGatherClosestv2<<<grid_size, block_size>>>(d_dist, d_output_dist, d_output_index, nb_yaals, nb_entities);
}


void cudaCollisionApplyv3(const float *d_input_x, const float *d_input_y, int *d_output_index, float *d_output_dist, int nb_yaals, int nb_entities) {
    // input size : nb_entities * 2
    // output size : nb_yaals * 2
    int block_size = 256;
    int grid_size = (nb_yaals + block_size - 1) / block_size;
    collisionGatherClosestv3<<<grid_size, block_size>>>(d_input_x, d_input_y, d_output_index, d_output_dist, nb_yaals, nb_entities);
    CUDA_CHECK_ERROR();
}

void cpuCollisionApply(const float *input_x, const float *input_y, float *output, int nb_yaals, int nb_entities) {
    // input size : nb_entities * 2
    // output size : nb_yaals * nb_entities
    for (int i = 0; i < nb_yaals; i++) {
        float min_dist = -1;
        int min_index = -1;
        for (int j = 0; j < nb_entities; j++) {
            float dx = input_x[j] - input_x[i];
            float dy = input_y[j] - input_y[i];
            float dist = dx * dx + dy * dy;
            if ((dist < min_dist || min_dist < 0) && dist > 0) {
                min_dist = dist;
                min_index = j;
            }
        }
        output[i] = min_dist;
    }
}

int main() {
    float *input_x, *input_y;
    int *output_index;
    float *output_dist_v1;
    float *output_dist_v2;
    float *output_dist_v3;
    float *output_dist_cpu;
    float *d_input_x, *d_input_y;
    float *d_dist, *d_dist_temp;
    int *d_indices, *d_indices_temp;
    int *d_output_index;
    float *d_output_dist_v1;
    float *d_output_dist_v2;
    float *d_output_dist_v3;
    int nb_yaals = 1;
    int log_nb_yaals = 23;
    int nb_entities = 2;
    int seed = 42;

    input_x = (float*)malloc(nb_entities * sizeof(float));
    input_y = (float*)malloc(nb_entities * sizeof(float));

    cudaMalloc((void**)&d_input_x, nb_entities * sizeof(float));
    cudaMalloc((void**)&d_input_y, nb_entities * sizeof(float));
    cudaMalloc((void**)&d_dist, nb_yaals * nb_entities * sizeof(float));
    cudaMalloc((void**)&d_dist_temp, nb_yaals * nb_entities * sizeof(float));
    cudaMalloc((void**)&d_indices, nb_yaals * nb_entities * sizeof(int));
    cudaMalloc((void**)&d_indices_temp, nb_yaals * nb_entities * sizeof(int));
    cudaMalloc((void**)&d_output_index, nb_yaals * sizeof(int));
    cudaMalloc((void**)&d_output_dist_v1, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_output_dist_v2, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_output_dist_v3, nb_yaals * sizeof(float));

    // Warmup
    cudaCollisionApplyv1(d_input_x, d_input_y, d_dist, d_output_dist_v1, d_output_index, nb_yaals, nb_entities);
    cudaCollisionApplyv2(d_input_x, d_input_y, d_dist, d_output_dist_v2, d_output_index, nb_yaals, nb_entities);
    cudaCollisionApplyv3(d_input_x, d_input_y, d_output_index, d_output_dist_v3, nb_yaals, nb_entities);
    cudaDeviceSynchronize();

    printf("\n\nWarmup done\n\n");

    nb_yaals = 2;
    nb_entities = 3;
    for (int i = 0; i < log_nb_yaals; i++) {
        nb_yaals = nb_yaals * 3 / 2;
        nb_entities = nb_entities * 3 / 2;

        // print output size
        printf("\n\ni : %d\n", i);
        printf("Nb yaals : %d, Nb entities : %d\n", nb_yaals, nb_entities);
        printf("Output size (in bytes): %d\n", nb_yaals * nb_entities * sizeof(float));

        input_x = (float*)malloc(nb_entities * sizeof(float));
        input_y = (float*)malloc(nb_entities * sizeof(float));
        cudaMalloc((void**)&d_input_x, nb_entities * sizeof(float));
        cudaMalloc((void**)&d_input_y, nb_entities * sizeof(float));
        cudaMalloc((void**)&d_dist, nb_yaals * nb_entities * sizeof(float));
        cudaMalloc((void**)&d_dist_temp, nb_yaals * nb_entities * sizeof(float));
        cudaMalloc((void**)&d_indices, nb_yaals * nb_entities * sizeof(int));
        cudaMalloc((void**)&d_indices_temp, nb_yaals * nb_entities * sizeof(int));
        cudaMalloc((void**)&d_output_index, nb_yaals * sizeof(float));
        cudaMalloc((void**)&d_output_dist_v1, nb_yaals * sizeof(float));
        cudaMalloc((void**)&d_output_dist_v2, nb_yaals * sizeof(float));
        cudaMalloc((void**)&d_output_dist_v3, nb_yaals * sizeof(float));

        // fill input
        srand(seed);
        for (int i = 0; i < nb_entities; i++) {
            input_x[i] = (float)rand() / (float)RAND_MAX;
            input_y[i] = (float)rand() / (float)RAND_MAX;
        }
        cudaMemcpy(d_input_x, input_x, nb_entities * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_y, input_y, nb_entities * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        float start = clock();
        cudaCollisionApplyv1(d_input_x, d_input_y, d_dist, d_output_dist_v1, d_output_index, nb_yaals, nb_entities);
        cudaDeviceSynchronize();
        float end = clock();
        printf("V1 time : %f\n", (end - start) / CLOCKS_PER_SEC);

        start = clock();
        cudaCollisionApplyv2(d_input_x, d_input_y, d_dist, d_output_dist_v2, d_output_index, nb_yaals, nb_entities);
        cudaDeviceSynchronize();
        end = clock();
        printf("V2 time : %f\n", (end - start) / CLOCKS_PER_SEC);

        start = clock();
        cudaCollisionApplyv3(d_input_x, d_input_y, d_output_index, d_output_dist_v3, nb_yaals, nb_entities);
        cudaDeviceSynchronize();
        end = clock();
        printf("V3 time : %f\n", (end - start) / CLOCKS_PER_SEC);

        output_dist_v1 = (float*)malloc(nb_yaals * sizeof(float));
        output_dist_v2 = (float*)malloc(nb_yaals * sizeof(float));
        output_dist_v3 = (float*)malloc(nb_yaals * sizeof(float));

        cudaMemcpy(output_dist_v1, d_output_dist_v1, nb_yaals * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(output_dist_v2, d_output_dist_v2, nb_yaals * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(output_dist_v3, d_output_dist_v3, nb_yaals * sizeof(float), cudaMemcpyDeviceToHost);

        output_dist_cpu = (float*)malloc(nb_yaals * sizeof(float));
        start = clock();
        cpuCollisionApply(input_x, input_y, output_dist_cpu, nb_yaals, nb_entities);
        end = clock();
        printf("CPU time : %f\n", (end - start) / CLOCKS_PER_SEC);

        int error_v1 = 0;
        int error_v2 = 0;
        int error_v3 = 0;

        for (int i = 0; i < nb_yaals; i++) {
            if ((output_dist_v1[i] - output_dist_cpu[i]) > 1e-8){
                error_v1++;
                //printf("Error at index %d : %f vs %f\n", i, output_dist_v1[i], output_dist_cpu[i]);
            }
            if ((output_dist_v2[i] - output_dist_cpu[i]) > 1e-8){
                error_v2++;
                //printf("Error at index %d : %f vs %f\n", i, output_dist_v2[i], output_dist_cpu[i]);
            }
            if ((output_dist_v3[i] - output_dist_cpu[i]) > 1e-8){
                error_v3++;
                //printf("Error at index %d : %f vs %f\n", i, output_dist_v3[i], output_dist_cpu[i]);
            }
        }
        printf("Nb Error in V1 : %d\n", error_v1);
        printf("Nb Error in V2 : %d\n", error_v2);
        printf("Nb Error in V3 : %d\n", error_v3);
    }

    // for (int i = 0; i < nb_yaals; i++) {
    //     printf("%f ", output_dist[i]);
    // }

    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_dist);
    cudaFree(d_dist_temp);
    cudaFree(d_indices);
    cudaFree(d_indices_temp);
    cudaFree(d_output_index);
    cudaFree(d_output_dist_v1);
    cudaFree(d_output_dist_v2);
    cudaFree(d_output_dist_v3);

    free(output_dist_v1);
    free(output_dist_v2);
    free(output_dist_v3);
    free(output_dist_cpu);
    free(input_x);
    free(input_y);

    return 0;
}