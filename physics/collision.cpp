#include <Eigen/Dense>

/**
 * given an array points of size (n 2) of points and a vector2f v, return the closest point to v
*/

/*
int closestIndex(const Eigen::MatrixXf& points, const Eigen::Vector2f& v) {
    Eigen::MatrixXf distances = (points.rowwise() - v.transpose()).rowwise().squaredNorm();
    // if some distance is 0, replace it with infinity
    distances = distances.array() == 0 ? std::numeric_limits<float>::infinity() : distances;

    int minIndex = distances.minCoeff();
    return minIndex;
}*/
