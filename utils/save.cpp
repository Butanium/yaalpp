#include "save.hpp"
#include "utils.h"
#include <fstream>
#include <iostream>


json data_to_json(const float *data, int size) {
    json j;
    for (int i = 0; i < size; i++) {
        j.push_back(data[i]);
    }
    return j;
}

void json_to_data(const json &johnson, float *data) {
    int i = 0;
    for (auto &el : johnson) {
        data[i] = el;
        i++;
    }
}

template<int nb_dims>
json tensor_to_json(const Tensor<float, nb_dims> &tensor) {
    json j;
    j["nb_dims"] = nb_dims;
    j["sizes"] = tensor.dimensions();
    j["data"] = data_to_json(tensor.data(), tensor.size());
    return j;
}

// TODO : omp
template<int nb_dims>
Tensor<float, nb_dims> json_to_tensor(const json &johnson) {
    std::vector<int> sizes = johnson["sizes"];
    if (sizes.size() != nb_dims) {
        throw std::invalid_argument("Incorrect number of dimensions.");
    }

    array<Index, nb_dims> shape;
    for (int i = 0; i < nb_dims; i++) {
        shape[i] = sizes[i];
    }

    Tensor<float, nb_dims> tensor(shape);
    json_to_data(johnson["data"], tensor.data());

    return tensor;
}

json filter_to_json(const SeparableFilter &filter) {
    json j;
    j["filter_size"] = filter.filter_size;
    j["nb_channels"] = filter.nb_channels;
    j["border_condition"] = filter.border_condition;
    j["skip_color_channels"] = filter.skip_color_channels;
    j["sigma"] = filter.sigma;
    return j;
}

SeparableFilter json_to_filter(const json &johnson) {
    int filter_size = johnson["filter_size"];
    int nb_channels = johnson["nb_channels"];
    int border_condition = johnson["border_condition"];
    bool skip_color_channels = johnson["skip_color_channels"];
    auto sigma_j = johnson["sigma"];
    std::vector<float> sigma;
    for (auto &el : sigma_j) {
        sigma.push_back(el);
    }

    SeparableFilter filter = SeparableFilter(filter_size, nb_channels, border_condition, skip_color_channels, std::move(sigma));

    return filter;
}

json plant_to_json(const Plant &plant) {
    json j;
    j["pos"] = {
            {"x", plant.position.x()},
            {"y", plant.position.y()}
    };
    // TODO?: De comment this if plant bodies are anything else than default
    //j["body"] = tensor_to_json(plant.body);
    return j;
}

Plant json_to_plant(const json &johnson, int num_channels) {
    Vec2 position(johnson["pos"]["x"], johnson["pos"]["y"]);
    // TODO?: De comment this if plant bodies are anything else than default
    //auto body = json_to_tensor<3>(johnson["body"]);
    Plant plant = Plant(num_channels);
    plant.position = position;
    return plant;
}

json yaal_to_json(const Yaal &yaal) {
    json j;
    j["position"] = {
            {"x", yaal.position.x()},
            {"y", yaal.position.y()}
    };
    // TODO : only store empreinte as body is generated from that
    j["body"] = tensor_to_json(yaal.body);
    j["genome"] = {
            {"brain", tensor_to_json(yaal.genome.brain.direction_weights)},
            {"max_speed", yaal.genome.max_speed},
            {"field_of_view", yaal.genome.field_of_view},
            {"size", yaal.genome.size},
            {"signature", data_to_json(yaal.genome.signature.data(), yaal.genome.signature.size())}
    };

    return j;
}

Yaal json_to_yaal(const json &johnson) {
    Vec2 position(johnson["position"]["x"], johnson["position"]["y"]);
    auto body = json_to_tensor<3>(johnson["body"]);
    auto brain = YaalMLP{
            .direction_weights = json_to_tensor<1>(johnson["genome"]["brain"]),
    };
    YaalGenome genome = {
            .brain = brain,
            .max_speed = johnson["genome"]["max_speed"],
            .field_of_view = johnson["genome"]["field_of_view"],
            .size = johnson["genome"]["size"],
            .signature = {}
    };

    genome.signature.reserve(brain.direction_weights.size());
    for (auto &el : johnson["genome"]["signature"]) {
        genome.signature.push_back(el);
    }

    return Yaal(position, genome, body);
}

void save_environment(const Environment &env, const std::string &path, bool save_map) {
    json j;
    j["width"] = env.width;
    j["height"] = env.height;
    j["channels"] = env.num_channels;
    j["offset_padding"] = {
            {"top", env.offset_padding.top},
            {"bottom", env.offset_padding.bottom},
            {"left", env.offset_padding.left},
            {"right", env.offset_padding.right}
    };
    j["offset_sharing"] = {
            {"top", env.offset_sharing.top},
            {"bottom", env.offset_sharing.bottom},
            {"left", env.offset_sharing.left},
            {"right", env.offset_sharing.right}
    };
    j["top_left_position"] = {
            {"x", env.top_left_position.x()},
            {"y", env.top_left_position.y()}
    };
    j["global_height"] = env.global_height;
    j["global_width"] = env.global_width;
    j["decay_factors"] = {
            {"nb_channels", env.decay_factors.dimension(2)},
            {"data", data_to_json(env.decay_factors.data(), env.decay_factors.size())}
    };
    j["max_values"] = {
            {"nb_channels", env.max_values.dimension(2)},
            {"data", data_to_json(env.max_values.data(), env.max_values.size())}
    };
    if (save_map) {
        std::string map_path = path + "map.json";
        j["map"] = map_path;
        json map_j = tensor_to_json(env.map);
        std::ofstream o(map_path);
        o << map_j << std::endl;
    } else {
        j["map"] = "Not saved.";
    }

    j["diffusion_filter"] = filter_to_json(env.diffusion_filter);

    json yaal_json;
    for (auto &yaal : env.yaals) {
        yaal_json.push_back(yaal_to_json(yaal));
    }
    j["yaals"] = yaal_json;

    json plant_json;
    for (auto &plant : env.plants) {
        plant_json.push_back(plant_to_json(plant));
    }
    j["plants"] = plant_json;
    std::ofstream o(path + "env.json");
    o << j << std::endl;
}

Environment load_environment(const std::string &path) {
    std::ifstream i(path + "env.json");
    json j;
    i >> j;
    int height = j["height"];
    int width = j["width"];
    int channels = j["channels"];

    int off_pad_top = j["offset_padding"]["top"];
    int off_pad_bottom = j["offset_padding"]["bottom"];
    int off_pad_left = j["offset_padding"]["left"];
    int off_pad_right = j["offset_padding"]["right"];
    int off_shar_top = j["offset_sharing"]["top"];
    int off_shar_bottom = j["offset_sharing"]["bottom"];
    int off_shar_left = j["offset_sharing"]["left"];
    int off_shar_right = j["offset_sharing"]["right"];

    Vec2i top_left_position(j["top_left_position"]["x"], j["top_left_position"]["y"]);

    int global_height = j["global_height"];
    int global_width = j["global_width"];

    int nb_channels = j["decay_factors"]["nb_channels"];
    float *decay_factors_data = new float[nb_channels];
    json_to_data(j["decay_factors"]["data"], decay_factors_data);
    Eigen::TensorMap<Tensor<float, 3>> decay_factors(decay_factors_data, array<Index, 3>{1, 1, nb_channels});

    float *max_values_data = new float[nb_channels];
    json_to_data(j["max_values"]["data"], max_values_data);
    Eigen::TensorMap<Tensor<float, 3>> max_values(max_values_data, array<Index, 3>{1, 1, nb_channels});

    auto diffusion_filter = json_to_filter(j["diffusion_filter"]);

    json yaal_json = j["yaals"];
    std::vector<Yaal> yaals;
    for (auto &yaal_j : yaal_json) {
        Yaal yaal = json_to_yaal(yaal_j);
        yaals.push_back(yaal);
    }

    json plant_json = j["plants"];
    std::vector<Plant> plants;
    for (auto &plant_j : plant_json) {
        Plant plant = json_to_plant(plant_j, channels);
        plants.push_back(plant);
    }

    std::string map_path = j["map"];
    if (map_path != "Not saved.") {
        std::ifstream map_i(map_path);
        json map_j;
        map_i >> map_j;
        Tensor<float, 3> map = json_to_tensor<3>(map_j);
        auto env = Environment(
                std::move(map),
                decay_factors,
                max_values,
                diffusion_filter,
                off_pad_top, off_pad_bottom, off_pad_left, off_pad_right,
                off_shar_top, off_shar_bottom, off_shar_left, off_shar_right,
                top_left_position,
                global_height, global_width,
                std::move(yaals),
                std::move(plants)
        );
        return env;
    } else {
        return Environment(
                height, width, channels,
                decay_factors,
                max_values,
                diffusion_filter,
                off_pad_top, off_pad_bottom, off_pad_left, off_pad_right,
                off_shar_top, off_shar_bottom, off_shar_left, off_shar_right,
                std::move(top_left_position),
                global_height, global_width,
                std::move(yaals),
                std::move(plants)
        );
    }
}
