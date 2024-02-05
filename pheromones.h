#ifndef PHEROMONES_H
#define PHEROMONES_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

void diffuse_pheromones(Tensor<float, 3> &map, float *diffusion_rate);

#endif // !PHEROMONES_H
