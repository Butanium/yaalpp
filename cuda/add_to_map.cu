#include <iostream>
#include <cuda.h>

#define CUDA_CHECK_ERROR() { cudaError_t err = cudaGetLastError(); \
                              if (err != cudaSuccess) { \
                                  printf("CUDA error: %s\n", cudaGetErrorString(err)); \
                                  exit(-1); \
                              } \
                            }

__global__ void yaalAddToMapKernel(
        float* map, int height, int width, int channels,
        int nb_yaals, float* positions_x, float* positions_y, float* sizes, float *bodies_imprint,
        int tl_offset_x, int tl_offset_y
) {
    int start_yaal = blockIdx.x * blockDim.x + threadIdx.x;

    for (int yaal = start_yaal; yaal < nb_yaals; yaal += blockDim.x * gridDim.x) {
        // get the index wrt the position
        float size = sizes[yaal];
        int x = (int) std::round(positions_x[yaal]) - tl_offset_x - (int) size / 2;
        int y = (int) std::round(positions_y[yaal]) - tl_offset_y - (int) size / 2;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float dx = (float) i - size / 2.f - 0.5f;
                float dy = (float) j - size / 2.f - 0.5f;
                float norm = dx * dx + dy * dy;
                if (norm <= size * size / 4) {
                    for (int c = 0; c < channels; c++) {
                        map[
                                // TODO : this order is 4 * faster than the other one (seul problème : je sais plus quel ordre :o)
                                (y + i) * width * channels + (x + j) * channels + c
                        ] += bodies_imprint[
                                c * nb_yaals + yaal
                        ];
                    }
                }
            }
        }
    }

}

void yaalUpdateApply(
        float* map, int height, int width, int channels,
        int nb_yaals, float* positions_x, float* positions_y, float* sizes, float *bodies_imprint,
        int tl_offset_x, int tl_offset_y
) {
    // NOTE : for some reason, when channels gets bigger, the time taken suddenly plummits if the number of threads is not reduced.
    int blockSize = 128;
    int numBlocks = min((nb_yaals + blockSize - 1) / blockSize, 64);

    yaalAddToMapKernel<<<numBlocks, blockSize>>>(
            map, height, width, channels,
            nb_yaals, positions_x, positions_y, sizes, bodies_imprint,
            tl_offset_x, tl_offset_y
    );
    CUDA_CHECK_ERROR();
}


int main() {
    int seed = 42;
    srand(seed);

    int height = 10000;
    int width = 10000;
    int channels = 4;
    int nb_yaals = 80000;
    float *map, *d_map;
    float *positions_x, *d_positions_x;
    float *positions_y, *d_positions_y;
    float *sizes, *d_sizes;
    float *bodies_imprint, *d_bodies_imprint;
    float max_speed = 1;
    float delta_t = 1;
    int tl_offset_x = 0;
    int tl_offset_y = 0;
    int size = 9;
    int fov = 10;

    map = (float*) malloc(height * width * channels * sizeof(float));
    positions_x = (float*) malloc(nb_yaals * sizeof(float));
    positions_y = (float*) malloc(nb_yaals * sizeof(float));
    sizes = (float*) malloc(nb_yaals * sizeof(float));
    bodies_imprint = (float*) malloc(nb_yaals * channels * sizeof(float));


    cudaMalloc((void**)&d_map, height * width * channels * sizeof(float));
    cudaMalloc((void**)&d_positions_x, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_positions_y, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_sizes, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_bodies_imprint, nb_yaals * channels * sizeof(float));

    for (int i = 0; i < nb_yaals; i++) {
        for (int c = 0; c < channels; c++) {
            // between 0 and 1
            bodies_imprint[i * channels + c] = (float) (rand() / (float) RAND_MAX);
        }
        sizes[i] = size;
        positions_x[i] = (float) (rand() % (width - fov * 2 - (int) sizes[i]) + fov + (int) sizes[i] / 2);
        positions_y[i] = (float) (rand() % (height - fov * 2 - (int) sizes[i]) + fov + (int) sizes[i] / 2);
    }

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                map[c * height * width + i * width + j] = (float) (rand() % ((int) 2e6) - 1e6) / 1e6;
            }
        }
    }

    cudaMemcpy(d_map, map, height * width * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_x, positions_x, nb_yaals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_y, positions_y, nb_yaals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes, nb_yaals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies_imprint, bodies_imprint, nb_yaals * channels * sizeof(float), cudaMemcpyHostToDevice);

    yaalUpdateApply(
            d_map, height, width, channels,
            nb_yaals, d_positions_x, d_positions_y, d_sizes, d_bodies_imprint,
            tl_offset_x, tl_offset_y
    );

    cudaDeviceSynchronize();

    float start = clock();
    yaalUpdateApply(
            d_map, height, width, channels,
            nb_yaals, d_positions_x, d_positions_y, d_sizes, d_bodies_imprint,
            tl_offset_x, tl_offset_y
    );
    cudaDeviceSynchronize();
    float end = clock();
    std::cout << "GPU Time : " << (end - start) / CLOCKS_PER_SEC << std::endl;

    // start = clock();
    // yaalUpdateCPU(
    //     map, height, width, channels,
    //     nb_yaals, brains, positions_x, positions_y, fovs, sizes,
    //     max_speed, delta_t,
    //     tl_offset_x, tl_offset_y
    // );
    // end = clock();
    // std::cout << "CPU Time : " << (end - start) / CLOCKS_PER_SEC << std::endl;

    cudaFree(d_map);
    cudaFree(d_positions_x);
    cudaFree(d_positions_y);
    cudaFree(d_sizes);
    cudaFree(d_bodies_imprint);

    free(map);
    free(positions_x);
    free(positions_y);
    free(sizes);
    free(bodies_imprint);
}