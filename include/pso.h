#pragma once
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include <algorithm>
#include "ihcp_model.h"

struct PSOResult {
    std::vector<float> best_position;
    float best_fitness;
    float best_mean_error;
};

class ParticleSwarmOptimizer {
private:
    int population_size;
    int num_iterations;
    float w, c1, c2;
    float lower_bound, upper_bound;

    struct Particle {
        std::vector<float> position;
        std::vector<float> velocity;
        std::vector<float> best_position;
        float best_fitness = std::numeric_limits<float>::max();
        float best_mean_error = std::numeric_limits<float>::max();
    };

public:
    ParticleSwarmOptimizer(int pop_size, int iter, float inertia, float c1, float c2, float lb, float ub);
    PSOResult optimize(const IHCPModel& model, int dimensions);
};