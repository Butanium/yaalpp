#include <iostream>
#include <utility>
#include "Environment.h"
#include "../physics/quadtree.hpp"
#include <set>
#include <boost/mpi.hpp>

using Constants::Yaal::MAX_FIELD_OF_VIEW;
using Constants::Yaal::MAX_SIZE;
using Eigen::array;
using Eigen::Index;
using Constants::Environment::FILTER_SIZE;
using Constants::Environment::SHARED_SIZE;
using Constants::MPI::MAP_TAG;
using Constants::MPI::YAAL_TAG;

Environment::Environment(int height, int width, int channels,
                         std::vector<float> &decay_factors_v,
                         std::vector<float> &_diffusion_factor,
                         std::vector<float> &max_values_v) :
        map(Tensor<float, 3>(height + 2 * MAX_FIELD_OF_VIEW, width + 2 * MAX_FIELD_OF_VIEW, channels)), height(height),
        width(width), num_channels(channels),
        offset_padding(
                {.top =  MAX_FIELD_OF_VIEW, .bottom =  MAX_FIELD_OF_VIEW, .left =  MAX_FIELD_OF_VIEW, .right =  MAX_FIELD_OF_VIEW}),
        diffusion_filter(SeparableFilter(FILTER_SIZE, channels, true, std::move(_diffusion_factor))),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>(decay_factors_v.data(), array<Index, 3>{1, 1, channels})),
        max_values(Eigen::TensorMap<Tensor<float, 3>>(max_values_v.data(), array<Index, 3>{1, 1, channels})) {
    map.setZero();

}

Environment::Environment(int height, int width,
                         int channels,
                         std::vector<float> &decay_factors_v,
                         std::vector<float> &_diffusion_factor,
                         std::vector<float> &max_values_v,
                         Vec2 &&_top_left_position,
                         int num_mpi_rows, int num_mpi_columns, MPI_Comm mpi_world) :
        height(height), width(width), num_channels(channels),
        top_left_position(std::move(_top_left_position)),
        diffusion_filter(SeparableFilter(FILTER_SIZE, channels, true, std::move(_diffusion_factor))),
        decay_factors(Eigen::TensorMap<Tensor<float, 3>>(decay_factors_v.data(), array<Index, 3>{1, 1, channels})),
        max_values(Eigen::TensorMap<Tensor<float, 3>>(max_values_v.data(), array<Index, 3>{1, 1, channels})),
        mpi_world(mpi_world) {
    MPI_Comm_rank(mpi_world, &mpi_rank);
    mpi_row = mpi_rank / num_mpi_columns;
    mpi_column = mpi_rank % num_mpi_columns;
    for (int i = 0; i < 8; i++) {
        neighbourhood.add(i, mpi_rank, mpi_row, mpi_column, num_mpi_rows, num_mpi_columns);
    }
    for (int i = 0; i < 4; i++) {
        if (neighbourhood[i] != MPI_PROC_NULL) {
            offset_sharing[i] =
                    SHARED_SIZE;
        } else {
            offset_padding[i] =
                    MAX_FIELD_OF_VIEW;
        }
    }

    map = Tensor<float, 3>(
            height + offset_padding.top + offset_padding.bottom + offset_sharing.top + offset_sharing.bottom,
            width + offset_padding.left + offset_padding.right + offset_sharing.left + offset_sharing.right,
            channels);
    map.setZero();

}

Environment::Environment(int height, int width, int channels, Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                         Eigen::TensorMap<Tensor<float, 3>>
                         max_values, const SeparableFilter &diffusion_filter, int offset_padding_top,
                         int offset_padding_bottom, int offset_padding_left, int offset_padding_right,
                         Vec2 top_left_position, std::vector<Yaal> yaals_, std::vector<Plant> plants_) :
        height(height), width(width), num_channels(channels),
        offset_padding(
                {.top = offset_padding_top, .bottom = offset_padding_bottom, .left = offset_padding_left, .right = offset_padding_right}),
        top_left_position(std::move(top_left_position)),
        diffusion_filter(diffusion_filter),
        yaals(std::move(yaals_)),
        plants(std::move(plants_)), decay_factors(decay_factors),
        max_values(max_values) {
    map = Tensor<float, 3>(height + offset_padding.vertical(), width + offset_padding.horizontal(), channels);
    map.setZero();
    for (auto &yaal: yaals) {
        add_to_map(yaal);
    }
    for (auto &plant: plants) {
        add_to_map(plant);
    }
}

Environment::Environment(Tensor<float, 3> &&map_,
                         Eigen::TensorMap<Tensor<float, 3>> decay_factors,
                         Eigen::TensorMap<Tensor<float, 3>> max_values,
                         const SeparableFilter &diffusion_filter,
                         int offset_padding_top,
                         int offset_padding_bottom,
                         int offset_padding_left,
                         int offset_padding_right,
                         Vec2 top_left_position,
                         std::vector<Yaal> yaals,
                         std::vector<Plant> plants) :
        map(std::move(map_)),
        height((int) map.dimension(0) - 2 * MAX_FIELD_OF_VIEW),
        width((int) map.dimension(1) - 2 * MAX_FIELD_OF_VIEW),
        num_channels((int) map.dimension(2)),
        offset_padding(
                {.top = offset_padding_top, .bottom = offset_padding_bottom, .left = offset_padding_left, .right = offset_padding_right}),
        top_left_position(std::move(top_left_position)),
        diffusion_filter(diffusion_filter),
        yaals(std::move(yaals)), plants(std::move(plants)),
        decay_factors(decay_factors),
        max_values(max_values) {
}


void Environment::add_plant(Plant &&plant) {
    add_to_map(plant);
    plants.push_back(std::move(plant));
}

void Environment::add_plant(const Plant &plant) {
    plants.push_back(plant);
    add_to_map(plant);
}

void Environment::add_yaal(Yaal &&yaal) {
    add_to_map(yaal);
    yaals.push_back(std::move(yaal));
}

void Environment::add_yaal(const Yaal &yaal) {
    yaals.push_back(yaal);
    add_to_map(yaal);
}

void Environment::create_yaals_and_plants(int num_yaal, int num_plant) {
    for (int i = 0; i < num_yaal; i++) {
        Yaal yaal = Yaal::random(num_channels);
        yaal.set_random_position(Vec2((float) MAX_SIZE / 2., (float) MAX_SIZE / 2.),
                                 Vec2(width - MAX_SIZE / 2., height - MAX_SIZE / 2.));
        yaal.position += top_left_position;
        add_yaal(yaal);
    }
    for (int i = 0; i < num_plant; i++) {
        Plant plant = Plant(num_channels);
        plant.set_random_position(Vec2((float) MAX_SIZE / 2., (float) MAX_SIZE / 2.),
                                  Vec2(width - MAX_SIZE / 2., height - MAX_SIZE / 2.));
        plant.position += top_left_position;
        add_plant(plant);
    }
}

std::tuple<int, int> Environment::pos_to_index(const Vec2 &pos) {
    Vec2 float_pos = (pos - top_left_position +
                      Vec2(offset_padding.left, offset_padding.top));
    int x = (int) std::round(float_pos.x());
    int y = (int) std::round(float_pos.y());
    return {y, x};
}

void Environment::add_to_map(const Plant &plant) {
    auto [i, j] = pos_to_index(
            plant.position - Vec2((float) plant.body.dimension(0) / 2.f, (float) plant.body.dimension(1) / 2.f));
    array<Index, 3> offsets = {i, j, 0};
    auto slice = map.slice(offsets, plant.body.dimensions());
#pragma omp critical
    {
        slice += plant.body;
    }
}

void Environment::add_to_map(const Yaal &yaal) {
    auto [i, j] = pos_to_index(yaal.top_left_position());
    array<Index, 3> offsets = {i, j, 0};
    auto slice = map.slice(offsets, yaal.body.dimensions());
#pragma omp critical
    {
        slice += yaal.body;
    };
}

void Environment::add_pheromone(const Yaal &yaal) {
    auto pheromone = yaal.genome.generate_pheromone();
    int pheromone_size = (int)pheromone.dimension(0);

    // Center the pheromone on the Yaal's position
    Vec2 pheromone_top_left = yaal.position - Vec2((float)pheromone_size / 2.0f, (float)pheromone_size / 2.0f);
    auto [i, j] = pos_to_index(pheromone_top_left);

    array<Index, 3> offsets = {i, j, 0};
    auto slice = map.slice(offsets, pheromone.dimensions());
#pragma omp critical
    {
        slice += pheromone;
    }
}

bool Environment::resolve_collisions(const std::vector<Vec2> &closests) {
    bool resolved = true;
#pragma omp parallel for schedule(static) shared(resolved)
    for (int i = 0; i < (int) yaals.size(); i++) {
        // resolve potential collision with other yaal
        Vec2 diff = closests[i] - yaals[i].position;
        float size = (float) yaals[i].genome.size / 2.f;
        float overlap = 2 * size - diff.norm();
        // TODO : when the sizes are not the same, put Yaals in quadtree, not only positions
        if (overlap > Constants::PHYSICS_EPSILON) {
            resolved = false;
            diff *= overlap / diff.norm();
            yaals[i].position -= diff / 2.f;
        }
        Vec2 relative_position = yaals[i].position - top_left_position;
        if (relative_position.x() < size || relative_position.y() < size ||
            relative_position.x() > (float) width - size || relative_position.y() > (float) height - size) {
            resolved = false;
            yaals[i].position = relative_position.cwiseMax(Vec2(size, size)).cwiseMin(
                    Vec2((float) width - size, (float) height - size)) + top_left_position;
        }
    }
    return resolved;
}

int Environment::handle_attacks() {
    int total_attacks = 0;

    // Build quadtree for efficient proximity queries
    if (yaals.size() < 2) return 0;

    QuadTree quadtree(Rect(top_left_position, Vec2(width, height)), (float) MAX_SIZE);
    quadtree.initialize(yaals);

    // Check each Yaal for nearby targets
    for (size_t i = 0; i < yaals.size(); i++) {
        auto &attacker = yaals[i];

        // Decide if this Yaal wants to attack
        std::uniform_real_distribution<float> attack_roll(0, 1);
        float attack_chance = Constants::Yaal::ATTACK_PROBABILITY * attacker.genome.aggressiveness;
        if (attack_roll(Yaal::generator) > attack_chance) {
            continue;  // This Yaal doesn't attack this timestep
        }

        // Find nearest Yaal
        Vec2 nearest_pos;
        if (!quadtree.get_closest(attacker.position, (float)attacker.genome.size, nearest_pos)) {
            continue;  // No nearby Yaal
        }

        // Find the defender by position
        for (size_t j = 0; j < yaals.size(); j++) {
            if (i == j) continue;

            auto &defender = yaals[j];
            if ((defender.position - nearest_pos).norm() < 0.1f) {
                // Found the defender! Calculate attack success
                float size_advantage = (float)(attacker.genome.size - defender.genome.size);
                float energy_advantage = attacker.energy - defender.energy;

                float success_chance = 0.5f;  // Base 50% success
                success_chance += size_advantage * Constants::Yaal::SIZE_ATTACK_BONUS;
                success_chance += energy_advantage * Constants::Yaal::ENERGY_ATTACK_BONUS;
                success_chance = std::clamp(success_chance, 0.1f, 0.9f);  // 10%-90% range

                if (attack_roll(Yaal::generator) < success_chance) {
                    // Attack succeeds! Steal energy
                    float energy_stolen = std::min(Constants::Yaal::ATTACK_BASE_ENERGY_STEAL, defender.energy);
                    defender.energy -= energy_stolen;
                    attacker.energy += energy_stolen;
                    attacker.energy = std::min(attacker.energy, attacker.genome.max_energy);
                    total_attacks++;
                }

                break;  // Only attack one Yaal per timestep
            }
        }
    }

    return total_attacks;
}

int Environment::consume_plants() {
    int plants_eaten = 0;
    std::vector<bool> plant_eaten(plants.size(), false);

    // Check each Yaal against each plant
    for (auto &yaal : yaals) {
        for (size_t p = 0; p < plants.size(); p++) {
            if (plant_eaten[p]) continue;

            float distance = (yaal.position - plants[p].position).norm();
            float yaal_radius = (float)yaal.genome.size / 2.0f;
            float plant_radius = (float)plants[p].body.dimension(0) / 2.0f;

            // If Yaal overlaps with plant, consume it
            if (distance < (yaal_radius + plant_radius)) {
                yaal.energy += Constants::Yaal::PLANT_ENERGY_GAIN;
                yaal.energy = std::min(yaal.energy, yaal.genome.max_energy);
                plant_eaten[p] = true;
                plants_eaten++;
            }
        }
    }

    // Remove eaten plants
    std::vector<Plant> surviving_plants;
    for (size_t i = 0; i < plants.size(); i++) {
        if (!plant_eaten[i]) {
            surviving_plants.push_back(std::move(plants[i]));
        }
    }
    plants = std::move(surviving_plants);

    return plants_eaten;
}

std::pair<int, int> Environment::handle_life_cycle() {
    int births = 0;
    int deaths = 0;
    std::vector<Yaal> new_yaals;
    std::vector<Yaal> surviving_yaals;

    for (auto &yaal : yaals) {
        // Check if Yaal dies (energy depleted or too old)
        if (yaal.energy <= 0.0f || yaal.age >= Constants::Yaal::MAX_AGE) {
            deaths++;
            continue;  // Don't add to surviving_yaals
        }

        // Check if Yaal reproduces
        if (yaal.energy > yaal.genome.max_energy * Constants::Yaal::REPRODUCTION_THRESHOLD) {
            // Split energy between parent and offspring
            float offspring_energy = yaal.energy / 2.0f;
            yaal.energy = offspring_energy;

            // Create offspring with mutated genome
            YaalGenome offspring_genome = yaal.genome;

            // Mutate the offspring genome
            std::uniform_real_distribution<float> mutation_chance(0, 1);
            std::uniform_real_distribution<float> mutation_delta(-Constants::Yaal::MUTATION_STRENGTH,
                                                                  Constants::Yaal::MUTATION_STRENGTH);

            if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                offspring_genome.max_speed *= (1.0f + mutation_delta(Yaal::generator));
                offspring_genome.max_speed = std::clamp(offspring_genome.max_speed,
                                                        Constants::Yaal::MIN_SPEED,
                                                        Constants::Yaal::MAX_SPEED);
            }

            if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                offspring_genome.max_energy *= (1.0f + mutation_delta(Yaal::generator));
                offspring_genome.max_energy = std::clamp(offspring_genome.max_energy,
                                                         Constants::Yaal::MIN_ENERGY,
                                                         Constants::Yaal::MAX_ENERGY);
            }

            if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                offspring_genome.energy_cost *= (1.0f + mutation_delta(Yaal::generator));
                offspring_genome.energy_cost = std::clamp(offspring_genome.energy_cost,
                                                          Constants::Yaal::MIN_ENERGY_COST,
                                                          Constants::Yaal::MAX_ENERGY_COST);
            }

            if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                offspring_genome.pheromone_intensity *= (1.0f + mutation_delta(Yaal::generator));
                offspring_genome.pheromone_intensity = std::clamp(offspring_genome.pheromone_intensity,
                                                                  Constants::Yaal::MIN_PHEROMONE_INTENSITY,
                                                                  Constants::Yaal::MAX_PHEROMONE_INTENSITY);
            }

            if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                offspring_genome.aggressiveness *= (1.0f + mutation_delta(Yaal::generator));
                offspring_genome.aggressiveness = std::clamp(offspring_genome.aggressiveness, 0.0f, 2.0f);
            }

            // Mutate brain weights
            for (int i = 0; i < offspring_genome.brain.direction_weights.size(); i++) {
                if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                    offspring_genome.brain.direction_weights(i) += mutation_delta(Yaal::generator);
                }
            }

            // Mutate signature (color/smell)
            for (size_t i = 0; i < offspring_genome.signature.size(); i++) {
                if (mutation_chance(Yaal::generator) < Constants::Yaal::MUTATION_RATE) {
                    offspring_genome.signature[i] += mutation_delta(Yaal::generator);
                    offspring_genome.signature[i] = std::clamp(offspring_genome.signature[i], 0.0f, 1.0f);
                }
            }

            // Create offspring near parent with random offset
            std::uniform_real_distribution<float> offset_dist(-10.0f, 10.0f);
            Vec2 offspring_pos = yaal.position + Vec2(offset_dist(Yaal::generator),
                                                      offset_dist(Yaal::generator));

            auto offspring_body = offspring_genome.generate_body();
            Yaal offspring(offspring_pos, std::move(offspring_genome), std::move(offspring_body));
            offspring.energy = offspring_energy;
            offspring.age = 0;

            new_yaals.push_back(std::move(offspring));
            births++;
        }

        surviving_yaals.push_back(std::move(yaal));
    }

    // Update yaals list
    yaals = std::move(surviving_yaals);
    for (auto &new_yaal : new_yaals) {
        yaals.push_back(std::move(new_yaal));
    }

    return {births, deaths};
}

int Environment::respawn_plants() {
    int spawned = 0;

    // Only spawn if below max capacity
    if ((int)plants.size() >= Constants::Environment::MAX_PLANTS_PER_AREA) {
        return 0;
    }

    // Probabilistic spawning
    std::uniform_real_distribution<float> chance(0, 1);
    if (chance(Plant::generator) < Constants::Environment::PLANT_RESPAWN_RATE) {
        Plant new_plant(num_channels);
        new_plant.set_random_position(Vec2((float) MAX_SIZE / 2., (float) MAX_SIZE / 2.),
                                      Vec2(width - MAX_SIZE / 2., height - MAX_SIZE / 2.));
        new_plant.position += top_left_position;
        add_plant(std::move(new_plant));
        spawned++;
    }

    return spawned;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedValue"

void Environment::mpi_sync() {
    // Send and receive map chunks to neighbours
    auto tot_offset = offset_padding + offset_sharing;
    auto recv_requests = std::vector<MPI_Request>();
    recv_requests.reserve(8);
    // @formatter:off
    const std::array<std::array<Index, 3>, 8> send_offsets = {{
        {{tot_offset.top, tot_offset.left, 0}},                          // top
        {{tot_offset.top + height - SHARED_SIZE, tot_offset.left, 0}},   // bottom
        {{tot_offset.top, tot_offset.left, 0}},                          // left
        {{tot_offset.top, tot_offset.left + width - SHARED_SIZE, 0}},    // right
        {{tot_offset.top, tot_offset.left, 0}},                          // top_left
        {{tot_offset.top, tot_offset.left + width - SHARED_SIZE, 0}},    // top_right
        {{tot_offset.top + height - SHARED_SIZE, tot_offset.left, 0}},   // bottom_left
        {{tot_offset.top + height - SHARED_SIZE, tot_offset.left + width - SHARED_SIZE, 0}} // bottom_right
    }};

    const array<array<Index, 3>, 8> recv_offsets = {{
        {{0, tot_offset.left, 0}},                                // top
        {{height + tot_offset.top, tot_offset.left, 0}},          // bottom
        {{tot_offset.top, 0, 0}},                                 // left
        {{tot_offset.top, tot_offset.left + width, 0}},           // right
        {{0, 0, 0}},                                              // top_left
        {{0, tot_offset.left + width, 0}},                        // top_right
        {{tot_offset.top + height, 0, 0}},                        // bottom_left
        {{tot_offset.top + height, tot_offset.left + width, 0}}   // bottom_right
    }};

    const array<array<Index, 3>, 8> share_dims = {{
        {{SHARED_SIZE, width, num_channels}},                  // top
        {{SHARED_SIZE, width, num_channels}},                  // bottom
        {{height, SHARED_SIZE, num_channels}},                 // left
        {{height, SHARED_SIZE, num_channels}},                 // right
        {{SHARED_SIZE, SHARED_SIZE, num_channels}},            // top_left
        {{SHARED_SIZE, SHARED_SIZE, num_channels}},            // top_right
        {{SHARED_SIZE, SHARED_SIZE, num_channels}},            // bottom_left
        {{SHARED_SIZE, SHARED_SIZE, num_channels}}             // bottom_right
    }};
    array<Tensor<float, 3>, 8> mpi_receive_results;
    array<Tensor<float, 3>, 8> mpi_sent_tensors;
    // @formatter:on
    int to_recv = 0;
    for (int i = 0; i < 8; i++) {
        int rank = neighbourhood[i];
        if (rank == MPI_PROC_NULL) {
            recv_requests[i] = MPI_REQUEST_NULL;
            continue;
        }
        to_recv++;
        mpi_sent_tensors[i] = map.slice(send_offsets[i], share_dims[i]);
        auto shared_size = share_dims[i][0] * share_dims[i][1] * share_dims[i][2];
        mpi_receive_results[i] = Tensor<float, 3>(share_dims[i]);
        assert((int) mpi_sent_tensors[i].size() == shared_size);
        MPI_Request send_request;
        MPI_Isend(mpi_sent_tensors[i].data(), (int) shared_size, MPI_FLOAT, rank, MAP_TAG, mpi_world,
                  &send_request);
    /*   TODO? this can be reverted back if leveraging omp is actually useful 
        MPI_Irecv(mpi_receive_results[i].data(), (int) shared_size, MPI_FLOAT, rank, MAP_TAG, mpi_world,
                 &recv_requests[i]); */
    }
    for (int i = 0; i < 8; i++) {
        if (neighbourhood[i] == MPI_PROC_NULL) {
            continue;
        }
        auto offset = recv_offsets[i];
        auto dims = share_dims[i];
        auto shared_size = dims[0] * dims[1] * dims[2];
        MPI_Recv(mpi_receive_results[i].data(), (int) shared_size, MPI_FLOAT, neighbourhood[i], MAP_TAG, mpi_world,
                 MPI_STATUS_IGNORE);
        auto received_tensor = mpi_receive_results[i];
        map.slice(offset, dims) = received_tensor;
    }

//TODO? This doesn't speed up the process for now but it could be useful if we want to do something else while waiting for the receives
/* #pragma omp parallel shared(recv_requests, mpi_receive_results, recv_offsets, share_dims, to_recv, std::cout, map, mpi_rank, neighbourhood, std::cerr)
    {
#pragma omp single nowait
        {
            while (to_recv > 0) {
                int completed_idx;
                MPI_Waitany(8, recv_requests.data(), &completed_idx, MPI_STATUS_IGNORE);
                if (completed_idx != MPI_UNDEFINED) {
                    // std::cout << "Process " << mpi_rank << " received map from " << neighbourhood[completed_idx]
                    //           << "\n";
                    to_recv--;
#pragma omp task default(none) shared(map, mpi_receive_results, recv_offsets, share_dims, neighbourhood, std::cerr) firstprivate(completed_idx)
                    {
                        auto offset = recv_offsets[completed_idx];
                        auto dims = share_dims[completed_idx];
                        auto received_tensor = mpi_receive_results[completed_idx];
                        map.slice(offset, dims) = received_tensor;
                    }
                } else {
                    std::cerr << "Process " << mpi_rank << " received MPI_UNDEFINED" << std::endl;
                    throw std::runtime_error("MPI_UNDEFINED received from waitany");
                }
            }
        }
    } */

    MPI_Barrier(mpi_world);

}

#pragma clang diagnostic pop

void Environment::step() {
    /* Evaluate the Yaals
     * Exchange the Yaals with the other processes to be able to resolve collisions
     * Resolve collisions
     * Consume plants
     * Handle death and reproduction
     * Decay and diffuse the map
     * Add the Yaals to the map
     * Exchange the Yaals that are now on the other side of the border as well as the shared map sections
     * add the yaals that crossed a border to the map
     * */
    mpi_sync();
    auto tot_offset = offset_padding + offset_sharing;
#pragma omp parallel for schedule(static)
    for (auto &yaal: yaals) {
        auto view = get_view(yaal);
        yaal.update(view);
    }

    // Solve collisions
    // TODO : if all the rest is done on GPU, reslove them with glouton n^2 on GPU, not quadtree on CPU
    for (int i = 0; i < 2; i++) {
        if (yaals.size() + plants.size() <= 1 || yaals.empty()) {
            break;
        }

        // initialize quadtree
        QuadTree quadtree(Rect(top_left_position, Vec2(width, height)),
                          (float) MAX_SIZE / 2.f);
        quadtree.initialize(yaals);
        quadtree.add_plants(plants);

        // get closest yaals and resolve collisions
        std::vector<Vec2> closests(yaals.size());
        quadtree.get_all_closest(yaals, closests);
        if (resolve_collisions(closests)) {
            break;
        }
    }

    // Handle attacks (energy stealing between Yaals)
    int attacks = handle_attacks();

    // Consume plants for energy
    int plants_eaten = consume_plants();

    // Handle death and reproduction
    auto [births, deaths] = handle_life_cycle();

    // Respawn plants to maintain ecosystem
    int plants_spawned = respawn_plants();

    // Print evolution statistics (only from rank 0 to avoid spam)
    if (mpi_rank == 0 && (births > 0 || deaths > 0 || plants_eaten > 0 || plants_spawned > 0 || attacks > 0)) {
        std::cout << "Evolution: " << yaals.size() << " yaals, "
                  << plants.size() << " plants | "
                  << "Births: " << births << ", Deaths: " << deaths
                  << ", Plants eaten: " << plants_eaten
                  << ", Plants spawned: " << plants_spawned
                  << ", Attacks: " << attacks << std::endl;
    }

    // TODO?: put this in diffusion filter to parallelize it
    // TODO?: use slicing to avoid useless operation on border
    map *= decay_factors.broadcast(array<int, 3>{height + tot_offset.vertical(),
                                                 width + tot_offset.horizontal(), 1});
    diffusion_filter.apply_inplace(map, tot_offset);

#pragma omp parallel for schedule(static)
    for (auto &yaal: yaals) {
        // TODO: remove this check for yaal in sub_env boudn and implement yaal passing
        Vec2 relative_pos = yaal.position - top_left_position;
        if (relative_pos.x() < MAX_SIZE / 2.f || relative_pos.y() < MAX_SIZE / 2.f ||
            relative_pos.x() > width - MAX_SIZE / 2.f || relative_pos.y() > height - MAX_SIZE / 2.f) {
            std::cout << "Skipping Yaal: " << yaal.position << " is out of sub env bounds" << std::endl;
            continue;
        }
        // Deposit pheromones before adding body (so body is on top visually)
        add_pheromone(yaal);
        add_to_map(yaal);
    }
#pragma omp parallel for schedule(static)
    for (auto &plant: plants) {
        add_to_map(plant);
    }

    // TODO?: parallelize it
    // TODO?: use slicing to avoid useless operation on border
    map = map.cwiseMin(
            max_values.broadcast(array<int, 3>{height + tot_offset.vertical(),
                                               width + tot_offset.horizontal(), 1}));
};
