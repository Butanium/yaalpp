#pragma once

#include <filesystem>
#include <vector>
#include <mpi.h>

void remove_files_in_directory(const std::filesystem::path &path);

void ensure_directory_exists(const std::filesystem::path &path);

struct Offset {
    // stores the offset of the top, bottom, left, and right of the map, for padding and shared bands
    int top;
    int bottom;
    int left;
    int right;

    int vertical() {
        return top + bottom;
    };

    int horizontal() {
        return left + right;
    };

    Offset operator+(const Offset &other) const {
        return {top + other.top, bottom + other.bottom, left + other.left, right + other.right};
    }


    static Offset zero() {
        return {0, 0, 0, 0};
    }

};

struct Neighbourhood {
    int top;
    int bottom;
    int left;
    int right;
    int top_left;
    int top_right;
    int bottom_left;
    int bottom_right;

    /// Subscript operator
    int &operator[](int i) {
        int *n_arr = &top;
        return n_arr[i];
    }

    static Neighbourhood none() {
        int n = MPI_PROC_NULL;
        return {n, n, n, n, n, n, n, n};
    }

};

std::vector<int> prime_decomposition(int n);

/**
 * Decompose the number of processes so they are arranged in a grid as close to a square as possible.
 * @param n number of processes
 * @param allow_discard if the decomposition is too far from squares, allow to discard some processes to get closer to squares
 * @param threshold if allow_discard is true, the maximum ratio between the two dimensions of the grid
 * @return
 */
std::tuple<int, int> grid_decomposition(int n, bool allow_discard = false, float threshold = 3);