#include "mpi_decomposition.hpp"

std::vector<int> prime_decomposition(int n) {
    std::vector<int> factors;
    int i = 2;
    while (i * i <= n) {
        if (n % i) {
            i += 1;
        } else {
            n /= i;
            factors.push_back(i);
        }
    }
    if (n > 1) factors.push_back(n);

    return factors;
}

std::tuple<int, int> grid_decomposition(int n, bool allow_discard, int threshold) {
    std::vector<int> factors = prime_decomposition(n);
    int p1 = 1;
    int p2;

    int p1_min = 1;
    int p2_min = n;

    float min_ratio = n;
    for (int i = 0; i < factors.size(); i++) {
        p1 *= factors[i];
        p2 = n / p1;

        if (std::max((float) p1 / p2, (float) p2 / p1) < min_ratio) {
            min_ratio = std::max((float) p1 / p2, (float) p2 / p1);
            p1_min = p1;
            p2_min = p2;
        }
    }

    if (allow_discard) {
        if (min_ratio <= threshold) {
            return std::make_tuple(p1_min, p2_min);
        }
        return grid_decomposition(n - 1);
    } else {
        return std::make_tuple(p1_min, p2_min);
    }
}