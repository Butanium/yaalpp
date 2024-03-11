#include <nlohmann/json.hpp>

#include "../simulation/Environment.h"

using json = nlohmann::json;

/* This file contains headers for the save and load functions for the environment and the yaals
 * */
/**
 * Convert a piece of contiguous data given as a pointer and a size to a json object
 */
json data_to_json(const float *data, int size);

/**
 * Convert a json object to a piece of contiguous data
 */
void json_to_data(const json &johnson, float *data);

/**
 * Convert an Eigen::Tensor with an arbitrary size to a json object
 */
template<int nb_dims>
json tensor_to_json(const Tensor<float, nb_dims> &tensor);

/**
 * Convert a json object to an Eigen::Tensor
 */
template<int nb_dims>
Tensor<float, nb_dims> json_to_tensor(const json &johnson);

/**
 * Convert a SeparableFilter to a json object
 */
json filter_to_json(const SeparableFilter &filter);

/**
 * Convert a json object to a SeparableFilter
 */
SeparableFilter json_to_filter(const json &johnson);

/**
 * Convert a plant to a json object
 */
json plant_to_json(const Plant &plant);

/**
 * Convert a json object to a plant
 */
Plant json_to_plant(const json &johnson);

/**
 * Convert a yaal to a json object
 */
json yaal_to_json(const Yaal &yaal);

/**
 * Convert a json object to a yaal
 */
Yaal json_to_yaal(const json &johnson);

/**
* Save the environment to a file
*/
void save_environment(const Environment &env, const std::string &path, bool save_map = false);

/**
* Load the environment from a file
*/
Environment load_environment(const std::string &path);