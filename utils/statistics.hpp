#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "../entity/Yaal.h"

/**
 * Statistics tracker for population dynamics and evolution metrics
 */
class Statistics {
public:
    struct PopulationSnapshot {
        int timestep;
        int population;
        int births;
        int deaths;
        int attacks;
        int plants_eaten;
        int plants_spawned;
        int plant_count;

        // Energy statistics
        float avg_energy;
        float min_energy;
        float max_energy;
        float total_energy;

        // Age statistics
        float avg_age;
        int min_age;
        int max_age;

        // Trait statistics
        float avg_speed;
        float avg_energy_cost;
        float avg_aggressiveness;
        float avg_pheromone_intensity;

        // Genetic diversity
        float trait_variance_speed;
        float trait_variance_energy;
        float trait_variance_aggressiveness;

        PopulationSnapshot() : timestep(0), population(0), births(0), deaths(0),
            attacks(0), plants_eaten(0), plants_spawned(0), plant_count(0),
            avg_energy(0), min_energy(0), max_energy(0), total_energy(0),
            avg_age(0), min_age(0), max_age(0),
            avg_speed(0), avg_energy_cost(0), avg_aggressiveness(0), avg_pheromone_intensity(0),
            trait_variance_speed(0), trait_variance_energy(0), trait_variance_aggressiveness(0) {}
    };

private:
    std::vector<PopulationSnapshot> history;
    std::string output_directory;
    int current_timestep;

    float calculate_variance(const std::vector<float>& values, float mean) {
        if (values.empty()) return 0.0f;
        float variance = 0.0f;
        for (float val : values) {
            float diff = val - mean;
            variance += diff * diff;
        }
        return variance / (float)values.size();
    }

public:
    Statistics(const std::string& output_dir = "./stats")
        : output_directory(output_dir), current_timestep(0) {}

    void record_snapshot(int timestep, const std::vector<Yaal>& yaals, int plant_count,
                        int births, int deaths, int attacks, int plants_eaten, int plants_spawned) {
        PopulationSnapshot snapshot;
        snapshot.timestep = timestep;
        snapshot.population = (int)yaals.size();
        snapshot.births = births;
        snapshot.deaths = deaths;
        snapshot.attacks = attacks;
        snapshot.plants_eaten = plants_eaten;
        snapshot.plants_spawned = plants_spawned;
        snapshot.plant_count = plant_count;

        if (yaals.empty()) {
            history.push_back(snapshot);
            return;
        }

        // Calculate energy statistics
        float total_energy = 0.0f;
        float min_energy = yaals[0].energy;
        float max_energy = yaals[0].energy;

        // Calculate age statistics
        int total_age = 0;
        int min_age = yaals[0].age;
        int max_age = yaals[0].age;

        // Collect traits for variance calculation
        std::vector<float> speeds;
        std::vector<float> energy_costs;
        std::vector<float> aggressiveness_values;

        float total_speed = 0.0f;
        float total_energy_cost = 0.0f;
        float total_aggressiveness = 0.0f;
        float total_pheromone = 0.0f;

        for (const auto& yaal : yaals) {
            // Energy
            total_energy += yaal.energy;
            min_energy = std::min(min_energy, yaal.energy);
            max_energy = std::max(max_energy, yaal.energy);

            // Age
            total_age += yaal.age;
            min_age = std::min(min_age, yaal.age);
            max_age = std::max(max_age, yaal.age);

            // Traits
            speeds.push_back(yaal.genome.max_speed);
            energy_costs.push_back(yaal.genome.energy_cost);
            aggressiveness_values.push_back(yaal.genome.aggressiveness);

            total_speed += yaal.genome.max_speed;
            total_energy_cost += yaal.genome.energy_cost;
            total_aggressiveness += yaal.genome.aggressiveness;
            total_pheromone += yaal.genome.pheromone_intensity;
        }

        int pop = (int)yaals.size();
        snapshot.avg_energy = total_energy / (float)pop;
        snapshot.min_energy = min_energy;
        snapshot.max_energy = max_energy;
        snapshot.total_energy = total_energy;

        snapshot.avg_age = (float)total_age / (float)pop;
        snapshot.min_age = min_age;
        snapshot.max_age = max_age;

        snapshot.avg_speed = total_speed / (float)pop;
        snapshot.avg_energy_cost = total_energy_cost / (float)pop;
        snapshot.avg_aggressiveness = total_aggressiveness / (float)pop;
        snapshot.avg_pheromone_intensity = total_pheromone / (float)pop;

        // Calculate variances (genetic diversity metrics)
        snapshot.trait_variance_speed = calculate_variance(speeds, snapshot.avg_speed);
        snapshot.trait_variance_energy = calculate_variance(energy_costs, snapshot.avg_energy_cost);
        snapshot.trait_variance_aggressiveness = calculate_variance(aggressiveness_values, snapshot.avg_aggressiveness);

        history.push_back(snapshot);
        current_timestep = timestep;
    }

    void write_to_csv(const std::string& filename = "evolution_stats.csv") {
        std::ofstream file(output_directory + "/" + filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open statistics file: " << filename << std::endl;
            return;
        }

        // Write header
        file << "timestep,population,births,deaths,attacks,plants_eaten,plants_spawned,plant_count,"
             << "avg_energy,min_energy,max_energy,total_energy,"
             << "avg_age,min_age,max_age,"
             << "avg_speed,avg_energy_cost,avg_aggressiveness,avg_pheromone_intensity,"
             << "var_speed,var_energy_cost,var_aggressiveness\n";

        // Write data
        for (const auto& snapshot : history) {
            file << snapshot.timestep << ","
                 << snapshot.population << ","
                 << snapshot.births << ","
                 << snapshot.deaths << ","
                 << snapshot.attacks << ","
                 << snapshot.plants_eaten << ","
                 << snapshot.plants_spawned << ","
                 << snapshot.plant_count << ","
                 << snapshot.avg_energy << ","
                 << snapshot.min_energy << ","
                 << snapshot.max_energy << ","
                 << snapshot.total_energy << ","
                 << snapshot.avg_age << ","
                 << snapshot.min_age << ","
                 << snapshot.max_age << ","
                 << snapshot.avg_speed << ","
                 << snapshot.avg_energy_cost << ","
                 << snapshot.avg_aggressiveness << ","
                 << snapshot.avg_pheromone_intensity << ","
                 << snapshot.trait_variance_speed << ","
                 << snapshot.trait_variance_energy << ","
                 << snapshot.trait_variance_aggressiveness << "\n";
        }

        file.close();
        std::cout << "Statistics written to " << output_directory << "/" << filename << std::endl;
    }

    void print_summary() {
        if (history.empty()) {
            std::cout << "No statistics recorded yet." << std::endl;
            return;
        }

        const auto& latest = history.back();
        std::cout << "\n=== Evolution Statistics (Timestep " << latest.timestep << ") ===" << std::endl;
        std::cout << "Population: " << latest.population << " yaals, " << latest.plant_count << " plants" << std::endl;
        std::cout << "Events: Births=" << latest.births << ", Deaths=" << latest.deaths
                  << ", Attacks=" << latest.attacks << ", Plants eaten=" << latest.plants_eaten << std::endl;
        std::cout << "Energy: avg=" << latest.avg_energy << ", min=" << latest.min_energy
                  << ", max=" << latest.max_energy << std::endl;
        std::cout << "Age: avg=" << latest.avg_age << ", min=" << latest.min_age
                  << ", max=" << latest.max_age << std::endl;
        std::cout << "Traits: speed=" << latest.avg_speed << " (var=" << latest.trait_variance_speed << ")"
                  << ", aggressiveness=" << latest.avg_aggressiveness << " (var=" << latest.trait_variance_aggressiveness << ")" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    const std::vector<PopulationSnapshot>& get_history() const {
        return history;
    }
};
