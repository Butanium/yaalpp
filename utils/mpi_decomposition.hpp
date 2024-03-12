#include <vector>

std::vector<int> prime_decomposition(int n);

/**
 * Decompose the number of processes so they are arranged in a grid as close to a square as possible.
 * @param n number of processes
 * @param allow_discard if the decomposition is too far from squares, allow to discard some processes to get closer to squares
 * @param threshold if allow_discard is true, the maximum ratio between the two dimensions of the grid
 * @return
 */
std::tuple<int, int> grid_decomposition(int n, bool allow_discard=false, int threshold=3);