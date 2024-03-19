/**
 * Yaals are stored as arrays, one for their brains, one for their position, one for their fov.
 *
 * Get View kernel:
 *  each thread is assigned to a yaal and has to retrieve the view in the map.
 *  Map accesses are not contiguous at all. See if it is worth it.
 *  Views are stored in a contiguous access array.
 *
 * Update kernel:
 *  first get the brain, stored in a way that makes them retrievable contiguously.
 *  then apply the brain to the view.
 *
 * No need to make it two kernels and store the views, can be done in one kernel without storage.
*/

#include <iostream>
#include <cuda.h>

#define CUDA_CHECK_ERROR() { cudaError_t err = cudaGetLastError(); \
                              if (err != cudaSuccess) { \
                                  printf("CUDA error: %s\n", cudaGetErrorString(err)); \
                                  exit(-1); \
                              } \
                            }

__global__ void yaalUpdateKernel(
        float* map, int height, int width, int channels,
        int nb_yaals, float* brains, float* positions_x, float* positions_y, int* fovs, float* sizes,
        float max_speed, float delta_t,
        int tl_offset_x, int tl_offset_y
) {
    int start_yaal = blockIdx.x * blockDim.x + threadIdx.x;

    for (int yaal = start_yaal; yaal < nb_yaals; yaal += blockDim.x * gridDim.x) {
        // get the index wrt the position
        int x = (int) std::round(positions_x[yaal]) - tl_offset_x;
        int y = (int) std::round(positions_y[yaal]) - tl_offset_y;
        int fov = (int) fovs[yaal];
        int size = (int) sizes[yaal];
        x -= fov + size / 2;
        y -= fov + size / 2;

        // apply the brain to the view
        float direction_x = 0;
        float direction_y = 0;
        float weight = 0;
        for (int i = 0; i < 2 * fov + size; i++) {
            for (int j = 0; j < 2 * fov + size; j++) {
                weight = 0;
                for (int c = 0; c < channels; c++) {
                    weight += map[
                                      (y + i) * width * channels + (x + j) * channels + c
                                      //c * height * width + (y + i) * width + (x + j)
                              ] * brains[yaal * channels + c];
                }
                float dx = i - fov - (float) size / 2.f;
                float dy = j - fov - (float) size / 2.f;
                float norm = sqrt(dx * dx + dy * dy);
                if (norm < 1e-6) {
                    norm = 1e-6;
                }
                direction_x += dx * weight / norm;
                direction_y += dy * weight / norm;
            }
        }

        // normalize the direction and apply it to the position
        float norm = sqrt(direction_x * direction_x + direction_y * direction_y);
        if (norm > 1e-6) {
            direction_x /= norm;
            direction_y /= norm;
            positions_x[yaal] += direction_x * max_speed * delta_t;
            positions_y[yaal] += direction_y * max_speed * delta_t;

            positions_x[yaal] = fminf(fmaxf(positions_x[yaal], fov + size / 2), width - fov - size / 2);
            positions_y[yaal] = fminf(fmaxf(positions_y[yaal], fov + size / 2), height - fov - size / 2);
        }
    }

}

void yaalUpdateApply(
        float* map, int height, int width, int channels,
        int nb_yaals, float* brains, float* positions_x, float* positions_y, int* fovs, float* sizes,
        float max_speed, float delta_t,
        int tl_offset_x, int tl_offset_y
) {
    // NOTE : for some reason, when channels gets bigger, the time taken suddenly plummits if the number of threads is not reduced.
    int blockSize = 128;
    int numBlocks = min((nb_yaals + blockSize - 1) / blockSize, 64);

    yaalUpdateKernel<<<numBlocks, blockSize>>>(
            map, height, width, channels,
            nb_yaals, brains, positions_x, positions_y, fovs, sizes,
            max_speed, delta_t,
            tl_offset_x, tl_offset_y
    );
    CUDA_CHECK_ERROR();
}

void yaalUpdateCPU(
        float* map, int height, int width, int channels,
        int nb_yaals, float* brains, float* positions_x, float* positions_y, int* fovs, float* sizes,
        float max_speed, float delta_t,
        int tl_offset_x, int tl_offset_y
) {
    for (int yaal = 0; yaal < nb_yaals; yaal++) {
        // get the index wrt the position
        int x = (int) std::round(positions_x[yaal]) - tl_offset_x;
        int y = (int) std::round(positions_y[yaal]) - tl_offset_y;
        int fov = (int) fovs[yaal];
        int size = (int) sizes[yaal];
        x -= fov + size / 2;
        y -= fov + size / 2;

        // apply the brain to the view
        float direction_x = 0;
        float direction_y = 0;
        float weight = 0;
        for (int i = 0; i < 2 * fov + size; i++) {
            for (int j = 0; j < 2 * fov + size; j++) {
                weight = 0;
                for (int c = 0; c < channels; c++) {
                    weight += map[
                                      c * height * width + (y + i) * width + (x + j)
                              ] * brains[c * nb_yaals + yaal];
                }
                float dx = i - fov - (float) size / 2.f;
                float dy = j - fov - (float) size / 2.f;
                float norm = sqrt(dx * dx + dy * dy);
                if (norm < 1e-6) {
                    norm = 1e-6;
                }
                direction_x += dx * weight / norm;
                direction_y += dy * weight / norm;
            }
        }

        // normalize the direction and apply it to the position
        float norm = sqrt(direction_x * direction_x + direction_y * direction_y);
        if (norm > 1e-6) {
            direction_x /= norm;
            direction_y /= norm;
            positions_x[yaal] += direction_x * max_speed * delta_t;
            positions_y[yaal] += direction_y * max_speed * delta_t;

            positions_x[yaal] = fminf(fmaxf(positions_x[yaal], fov + size / 2), width - fov - size / 2);
            positions_y[yaal] = fminf(fmaxf(positions_y[yaal], fov + size / 2), height - fov - size / 2);
        }
    }
}

int main() {
    int seed = 42;
    srand(seed);

    int height = 1000;
    int width = 1000;
    int channels = 4;
    int nb_yaals = 80000;
    float *map, *d_map;
    float *brains, *d_brains;
    float *positions_x, *d_positions_x;
    float *positions_y, *d_positions_y;
    int *fovs, *d_fovs;
    float *sizes, *d_sizes;
    float max_speed = 1;
    float delta_t = 1;
    int tl_offset_x = 0;
    int tl_offset_y = 0;
    int fov = 10;
    int size = 9;

    map = (float*) malloc(height * width * channels * sizeof(float));
    brains = (float*) malloc(nb_yaals * channels * sizeof(float));
    positions_x = (float*) malloc(nb_yaals * sizeof(float));
    positions_y = (float*) malloc(nb_yaals * sizeof(float));
    fovs = (int*) malloc(nb_yaals * sizeof(int));
    sizes = (float*) malloc(nb_yaals * sizeof(float));

    cudaMalloc((void**)&d_map, height * width * channels * sizeof(float));
    cudaMalloc((void**)&d_brains, nb_yaals * channels * sizeof(float));
    cudaMalloc((void**)&d_positions_x, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_positions_y, nb_yaals * sizeof(float));
    cudaMalloc((void**)&d_fovs, nb_yaals * sizeof(int));
    cudaMalloc((void**)&d_sizes, nb_yaals * sizeof(float));

    for (int i = 0; i < nb_yaals; i++) {
        for (int c = 0; c < channels; c++) {
            // between -1 and 1
            brains[c * nb_yaals + i] = (float) (rand() % ((int) 2e6) - 1e6) / 1e6;
        }
        fovs[i] = fov;
        sizes[i] = size;
        positions_x[i] = (float) (rand() % (width - fovs[i] * 2 - (int) sizes[i]) + fovs[i] + (int) sizes[i] / 2);
        positions_y[i] = (float) (rand() % (height - fovs[i] * 2 - (int) sizes[i]) + fovs[i] + (int) sizes[i] / 2);
    }

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                map[c * height * width + i * width + j] = (float) (rand() % ((int) 2e6) - 1e6) / 1e6;
            }
        }
    }

    cudaMemcpy(d_map, map, height * width * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_brains, brains, nb_yaals * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_x, positions_x, nb_yaals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_y, positions_y, nb_yaals * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fovs, fovs, nb_yaals * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes, nb_yaals * sizeof(float), cudaMemcpyHostToDevice);

    yaalUpdateApply(
            d_map, height, width, channels,
            nb_yaals, d_brains, d_positions_x, d_positions_y, d_fovs, d_sizes,
            max_speed, delta_t,
            tl_offset_x, tl_offset_y
    );

    cudaDeviceSynchronize();

    float start = clock();
    yaalUpdateApply(
            d_map, height, width, channels,
            nb_yaals, d_brains, d_positions_x, d_positions_y, d_fovs, d_sizes,
            max_speed, delta_t,
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
    cudaFree(d_brains);
    cudaFree(d_positions_x);
    cudaFree(d_positions_y);
    cudaFree(d_fovs);
    cudaFree(d_sizes);

    free(map);
    free(brains);
    free(positions_x);
    free(positions_y);
    free(fovs);
    free(sizes);
}