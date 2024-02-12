#include "pheromones.h"

#define SAFE_SELECT(tensor, x, y, c) (x >= 0 && x < tensor.dimension(0) && y >= 0 && y < tensor.dimension(1) ? tensor(x, y, c) : 0)

template <typename T>
T apply_diffusion_kernel(Tensor<T, 3> &map, int x, int y, int c, float diffusion_rate) {
  float side_rate = diffusion_rate / 9;
  float center_rate = 1 - 8 * side_rate;
  return center_rate * map(x, y, c) + side_rate * (SAFE_SELECT(map, x-1, y, c) + SAFE_SELECT(map, x+1, y, c) + SAFE_SELECT(map, x, y-1, c) +
    SAFE_SELECT(map, x, y+1, c) + SAFE_SELECT(map, x-1, y-1, c) + SAFE_SELECT(map, x+1, y-1, c) + SAFE_SELECT(map, x-1, y+1, c) + SAFE_SELECT(map, x+1, y+1, c));
}

void diffuse_pheromones(Tensor<float, 3> &map, float *diffusion_rate) {
  for (int x = 0; x < map.dimension(0); x++) {
    for (int y = 0; y < map.dimension(1); y++) {
      for (int c = 0; c < map.dimension(2); c++) {
        map(x, y, c) = apply_diffusion_kernel(map, x, y, c, diffusion_rate[c]);
      }
    }
  }
}
