#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    // Create a (100, 100, 5) array of random values
    using Eigen::Tensor;
    using Eigen::array;
    int height = 3;
    int width = 3;
    int channels = 2;
    Eigen::Tensor<double, 3> A(height, width, channels);
    A.setRandom();
    std::cout << "The array A is:\n" << A.format(Eigen::TensorIOFormat::Numpy()) << std::endl;

    // Create a (5) array of random values
    Eigen::Tensor<double, 1> B(channels);
    B.setRandom();
    std::cout << "The array B is:\n" << B.format(Eigen::TensorIOFormat::Numpy()) << std::endl;
    Tensor<float, 2> input(7, 11);
    array<int, 3> three_dims{{7, 11, 1}};
    Tensor<float, 3> result = input.reshape(three_dims);
    auto C =
            B.reshape(Eigen::array < Eigen::Index, 3 > {1, 1, channels})
                    .broadcast(Eigen::array < Eigen::Index, 3 > {height, width, 1});
    // if you use Tensor<float, 3> C = ... instead of auto C = ..., that will create a new tensor instead of a reference
    B(0) = 1;
    std::cout << "The array B is:\n" << B.format(Eigen::TensorIOFormat::Numpy()) << std::endl;
    std::cout << "The array C is:\n" << C.format(Eigen::TensorIOFormat::Numpy()) << std::endl;

    // Multiply the arrays element-wise on the last dimension
    A *= C;

    // Print the result
    std::cout << "The result of the element-wise multiplication is:\n" << A.format(Eigen::TensorIOFormat::Plain())
              << std::endl;

    return 0;
}