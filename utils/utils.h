#pragma once

#include <filesystem>
#include <vector>
#include <mpi.h>

void remove_files_in_directory(const std::filesystem::path &path);

void ensure_directory_exists(const std::filesystem::path &path);

void remove_directory_recursively(const std::filesystem::path &path);

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

    /// Subscript operator
    int &operator[](int i) {
        int *n_arr = &top;
        return n_arr[i];
    }

    friend std::ostream &operator<<(std::ostream &os, const Offset &o) {
        os << "Offset(top: " << o.top << ", bottom: " << o.bottom << ", left: " << o.left << ", right: " << o.right
           << ")";
        return os;
    }

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

    friend std::ostream &operator<<(std::ostream &os, const Neighbourhood &n) {
        os << "Neighbourhood(top: " << n.top << ", bottom: " << n.bottom << ", left: " << n.left << ", right: "
           << n.right
           << ", top_left: " << n.top_left << ", top_right: " << n.top_right << ", bottom_left: " << n.bottom_left
           << ", bottom_right: " << n.bottom_right << ")";
        return os;
    }

    static Neighbourhood none() {
        int n = MPI_PROC_NULL;
        return {n, n, n, n, n, n, n, n};
    }

    bool add(int i, int mpi_rank, int row, int column, int num_rows, int num_columns) {
        int rank = mpi_rank;
        switch (i) {
            case 0:  // top
                if (row == 0) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank -= num_columns;
                }
                break;
            case 1:  // bottom
                if (row == num_rows - 1) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank += num_columns;
                }
                break;
            case 2: // left
                if (column == 0) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank -= 1;
                }
                break;
            case 3:  // right
                if (column == num_columns - 1) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank += 1;
                }
                break;
            case 4:  // top_left
                if (row == 0 || column == 0) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank -= num_columns + 1;
                }
                break;
            case 5:  // top_right
                if (row == 0 || column == num_columns - 1) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank -= num_columns - 1;
                }
                break;
            case 6:
                if (row == num_rows - 1 || column == 0) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank += num_columns - 1;
                }
                break;
            case 7:
                if (row == num_rows - 1 || column == num_columns - 1) {
                    rank = MPI_PROC_NULL;
                } else {
                    rank += num_columns + 1;
                }
                break;
            default:
                throw std::runtime_error("Invalid neighbourhood index");
        }
        (*this)[i] = rank;
        return rank != MPI_PROC_NULL;

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