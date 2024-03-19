//
// Created by clementd on 29/02/24.
//

#ifndef YAALPP_UTILS_H
#define YAALPP_UTILS_H

#include <filesystem>

void remove_files_in_directory(const std::filesystem::path &path);

void ensure_directory_exists(const std::filesystem::path &path);

void remove_directory_recursively(const std::filesystem::path &path);

#endif //YAALPP_UTILS_H
