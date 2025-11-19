#pragma once

namespace Constants {
    static constexpr float DELTA_T = 1;
    static constexpr float EPSILON = 1e-6;
    static constexpr float PHYSICS_EPSILON = 1e-2;
    namespace Yaal {
        static constexpr float MIN_SPEED = 0.1f;
        static constexpr float MAX_SPEED = 1.0f;
        static constexpr int MIN_FIELD_OF_VIEW = 1;
        static constexpr int MAX_FIELD_OF_VIEW = 3;
        static constexpr int MIN_SIZE = 9;
        static constexpr int MAX_SIZE = 9;
        static constexpr float MIN_ENERGY = 50.0f;
        static constexpr float MAX_ENERGY = 200.0f;
        static constexpr float MIN_ENERGY_COST = 0.05f;
        static constexpr float MAX_ENERGY_COST = 0.2f;
        static constexpr float REPRODUCTION_THRESHOLD = 0.75f;  // Reproduce when energy > 75% of max
        static constexpr float PLANT_ENERGY_GAIN = 20.0f;  // Energy gained from eating a plant
        static constexpr float MUTATION_RATE = 0.1f;  // 10% chance of mutation per parameter
        static constexpr float MUTATION_STRENGTH = 0.1f;  // Mutations are +/- 10% of current value
        static constexpr float PHEROMONE_STRENGTH = 0.3f;  // Strength of pheromone deposit per timestep
        static constexpr float MIN_PHEROMONE_INTENSITY = 0.0f;
        static constexpr float MAX_PHEROMONE_INTENSITY = 1.0f;
        static constexpr int PRIME_AGE = 500;  // Age at which Yaals are at peak performance
        static constexpr int MAX_AGE = 2000;  // Maximum age before death from old age
        static constexpr float AGE_SPEED_PENALTY = 0.0005f;  // Speed reduction per timestep after prime
        static constexpr float AGE_ENERGY_PENALTY = 0.0003f;  // Energy efficiency loss per timestep after prime
    }
    namespace Environment {
        static constexpr int FILTER_SIZE = 7;
        static constexpr int SHARED_SIZE = Yaal::MAX_FIELD_OF_VIEW + Yaal::MAX_SIZE / 2 + 1;
        static constexpr float PLANT_RESPAWN_RATE = 0.05f;  // 5% chance per step to spawn a new plant
        static constexpr int MAX_PLANTS_PER_AREA = 50;  // Maximum plants per MPI process area
    }
    namespace MPI {
        static constexpr int MAP_TAG = 0;
        static constexpr int YAAL_TAG = 1;
    }
}