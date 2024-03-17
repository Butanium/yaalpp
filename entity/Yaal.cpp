//
// Created by clementd on 31/01/24.
//

#include "Yaal.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>


using Eigen::Tensor;
using Eigen::array;
using Vec2 = Eigen::Vector2f;

Tensor<float, 3> direction_matrix(int height, int width) {
    // Compute the direction matrix s.t. D_ij is the normalized direction from the center to the i,j pixel
    int dims[] = {height, width};
    Tensor<float, 3> directions(height, width, 2);
    directions.setZero();
    for (int d = 0; d < 2; d++) {
        float center = (float) dims[d] / 2.f - 0.5f;
        for (int i = 0; i < dims[d]; i++) {
            directions.chip(d, 2).chip(i, 1 - d).setConstant((float) (i - center));
        }
    }
    Tensor<float, 3> d_norms = directions.square().sum(array<Eigen::Index, 1>{2}).sqrt().reshape(
            array<Eigen::Index, 3>{height, width, 1}).broadcast(
            array<Eigen::Index, 3>{1, 1, 2});
    if (height % 2 && width % 2) {
        d_norms.chip(height / 2, 0).chip(width / 2, 0).setConstant(1);
    }
    return directions / d_norms;
}

void Yaal::bound_position(const Vec2 &min, const Vec2 &max) {
    position = position.cwiseMax(min).cwiseMin(max);
}

Yaal::Yaal(Vec2 &&position, YaalGenome &&genome, Tensor<float, 3> &&body) :
        position(std::move(position)),
        genome(std::move(genome)),
        body(std::move(body)) {}

Yaal::Yaal(const Vec2 &position, const YaalGenome &genome, const Tensor<float, 3> &body) :
        position(position),
        genome(genome), body(body) {}

std::mt19937 YaalGenome::generator = std::mt19937(std::random_device{}());
std::mt19937 Yaal::generator = std::mt19937(std::random_device{}());

/**
 * Generate a body with a given signature
 * @param size The size of the body
 * @param signature The signature of the yaal (i.e. the color and smell)
 */
Tensor<float, 3> YaalGenome::generate_body() {
    Tensor<float, 3> body(size, size, (long) signature.size());
    for (int c = 0; c < (int) signature.size(); c++)
        body.chip(c, 2).setConstant(signature[c]);
    // Apply circle mask
    float center = (float) size / 2.f - 0.5f;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float dx = (float) i - center;
            float dy = (float) j - center;
            auto slice = body.chip(i, 0).chip(j, 0);
            if (dx * dx + dy * dy > (float) (size * size) / 4.f) {
                slice.setZero();
            } else {
                // Interpolate between *= 0.5 and *= 1
                slice = slice * (0.5f + 0.5f * (1 - std::sqrt(dx * dx + dy * dy) / ((float) size / 2.f)));
            }
        }
    }
    return body;
}

template<typename Scalar>
struct GenomeEigenRandomGenerator {
    std::mt19937 &generator;

    // Default and copy constructors. Both are needed
    GenomeEigenRandomGenerator() : generator(YaalGenome::generator) {}

    GenomeEigenRandomGenerator(const GenomeEigenRandomGenerator &other) : generator(other.generator) {}

    // Return a random value to be used.  "element_location" is the
    // location of the entry to set in the tensor, it can typically
    // be ignored.
    Scalar operator()(Eigen::DenseIndex element_location,
                      Eigen::DenseIndex /*unused*/ = 0) const {
        std::uniform_real_distribution<Scalar> dist(0, 1);
        return dist(generator);
    }

    // Same as above but generates several numbers at a time.
    typename Eigen::internal::packet_traits<Scalar>::type packetOp(
            Eigen::DenseIndex packet_location, Eigen::DenseIndex /*unused*/ = 0) const {
        std::uniform_real_distribution<Scalar> dist(0, 1);
        return Eigen::internal::packet_traits<Scalar>::setConstant(dist(generator));
    }
};

YaalGenome YaalGenome::random(int num_channels) {
    auto speed_rng = std::uniform_real_distribution<float>(Constants::Yaal::MIN_SPEED, Constants::Yaal::MAX_SPEED);
    auto fov_rng = std::uniform_int_distribution<int>(Constants::Yaal::MIN_FIELD_OF_VIEW,
                                                      Constants::Yaal::MAX_FIELD_OF_VIEW);
    auto size_rng = std::uniform_int_distribution<int>(Constants::Yaal::MIN_SIZE, Constants::Yaal::MAX_SIZE);
    auto signature_rng = std::uniform_real_distribution<float>(0, 1);
    int size = size_rng(generator);
    std::vector<float> signature = std::vector<float>(num_channels);
    for (int i = 0; i < num_channels; i++) {
        signature[i] = signature_rng(generator);
    }
    return {
            .brain = YaalMLP{
                    .direction_weights = Tensor<float, 1>(num_channels).setRandom<GenomeEigenRandomGenerator<float>>()
            },
            .max_speed = speed_rng(generator),
            .field_of_view = fov_rng(generator),
            .size = size,
            .signature = signature
    };
}

Yaal Yaal::random(int num_channels, const Vec2 &position) {
    auto genome = YaalGenome::random(num_channels);
    return {position, genome, genome.generate_body()};
}

void Yaal::set_random_position(const Vec2 &min, const Vec2 &max) {
    std::uniform_real_distribution<float> x_rng(min.x(), max.x());
    std::uniform_real_distribution<float> y_rng(min.y(), max.y());
    position = {x_rng(generator), y_rng(generator)};
}

Yaal Yaal::random(int num_channels) {
    return random(num_channels, Vec2::Zero());
}

Vec2 Yaal::top_left_position() const {
    return position - Vec2((float) genome.size / 2.f, (float) genome.size / 2.f);
}
