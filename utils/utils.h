//
// Created by clementd on 29/02/24.
//

#pragma once

#include <filesystem>
#include <vector>

void remove_files_in_directory(const std::filesystem::path &path);

void ensure_directory_exists(const std::filesystem::path &path);

struct Offset {
    // stores the offset of the top, bottom, left, and right of the map, for padding and shared bands
    int top;
    int bottom;
    int left;
    int right;

    Offset operator+(const Offset &other) const {
        return {top + other.top, bottom + other.bottom, left + other.left, right + other.right};
    }
    static Offset zero() {
        return {0, 0, 0, 0};
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
std::tuple<int, int> grid_decomposition(int n, bool allow_discard = false, int threshold = 3);