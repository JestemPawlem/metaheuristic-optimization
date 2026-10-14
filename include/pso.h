#pragma once
#include <cstddef>
#include <vector>
#include <limits>

#include "ihcp_model.h"


struct PSOResult
{
    std::vector<float> best_position{};
    float best_fitness = std::numeric_limits<float>::max();
    float best_mean_error = std::numeric_limits<float>::max();
};

class ParticleSwarmOptimizer
{
public:
    ParticleSwarmOptimizer(
        std::size_t pop_size,
        std::size_t iter,
        float inertia,
        float c1,
        float c2,
        const std::vector<float>& lower_bounds,
        const std::vector<float>& upper_bounds);
    PSOResult optimize(const IHCPModel& model, std::size_t dimensions);

private:
    std::size_t population_size;
    std::size_t n_iterations;
    float inertia;
    float c1;
    float c2;
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;

    struct Particle
    {
        std::vector<float> position;
        std::vector<float> velocity;
        std::vector<float> best_position;
        float best_fitness = std::numeric_limits<float>::max();
        float best_mean_error = std::numeric_limits<float>::max();
    };
};