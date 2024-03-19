//
// Created by clementd on 29/02/24.
//
#include "utils.h"

void ensure_directory_exists(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }
}

void remove_files_in_directory(const std::filesystem::path &path) {
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry)) {
            std::filesystem::remove(entry);
        }
    }
}

void remove_directory_recursively(const std::filesystem::path &path) {
    if (std::filesystem::exists(path)) {
        std::filesystem::remove_all(path);
    }
}
