#include <algorithm>
#include <limits>
#include <print>
#include <random>

#include "color.h"
#include "pso.h"


ParticleSwarmOptimizer::ParticleSwarmOptimizer(
    std::size_t population_size,
    std::size_t n_iterations,
    float inertia,
    float c1,
    float c2,
    const std::vector<float>& lb,
    const std::vector<float>& ub) :
    population_size(population_size),
    n_iterations(n_iterations),
    inertia(inertia),
    c1(c1),
    c2(c2),
    lower_bounds(lb),
    upper_bounds(ub)
{}


PSOResult ParticleSwarmOptimizer::optimize(const IHCPModel& model, std::size_t dimensions) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::uniform_real_distribution<float>> dist_pos;
    std::vector<std::uniform_real_distribution<float>> dist_vel;
    std::vector<float> vmax(dimensions);

    for (std::size_t d{}; d < dimensions; ++d) {
        float range = upper_bounds[d] - lower_bounds[d];
        dist_pos.emplace_back(lower_bounds[d], upper_bounds[d]);
        dist_vel.emplace_back(-range * 0.05f, range * 0.05f);
        vmax[d] = range * 0.2f;
    }
    std::uniform_real_distribution<float> dist_rand(0.0f, 1.0f);

    std::vector<Particle> population(population_size);
    std::vector<float> global_best_position(dimensions);
    float global_best_fitness = std::numeric_limits<float>::max();
    float global_best_mean_error = std::numeric_limits<float>::max();

    for (std::size_t i{}; i < population_size; ++i) {
        population[i].position.resize(dimensions);
        population[i].velocity.resize(dimensions);
        population[i].best_position.resize(dimensions);

        for (int d = 0; d < dimensions; ++d) {
            population[i].position[d] = dist_pos[d](gen);
            population[i].velocity[d] = dist_vel[d](gen);
            population[i].best_position[d] = population[i].position[d];
        }

        OptimizationResult res = model.objective_function(population[i].position);
        population[i].best_fitness = res.fitness;
        population[i].best_mean_error = res.mean_error;

        if (res.fitness < global_best_fitness) {
            global_best_fitness = res.fitness;
            global_best_mean_error = res.mean_error;
            global_best_position = population[i].position;
        }
    }

    for (std::size_t iter{}; iter < n_iterations; ++iter) {
        for (std::size_t p{}; p < population_size; ++p) {
            for (std::size_t d{}; d < dimensions; ++d) {
                float r1 = dist_rand(gen);
                float r2 = dist_rand(gen);

                population[p].velocity[d] = inertia * population[p].velocity[d] +
                    c1 * r1 * (population[p].best_position[d] - population[p].position[d]) +
                    c2 * r2 * (global_best_position[d] - population[p].position[d]);

                population[p].velocity[d] = std::clamp(population[p].velocity[d], -vmax[d], vmax[d]);
                population[p].position[d] += population[p].velocity[d];

                if (population[p].position[d] < lower_bounds[d]) {
                    population[p].position[d] = lower_bounds[d];
                    population[p].velocity[d] = 0.0f;
                }
                else if (population[p].position[d] > upper_bounds[d]) {
                    population[p].position[d] = upper_bounds[d];
                    population[p].velocity[d] = 0.0f;
                }
            }

            OptimizationResult res = model.objective_function(population[p].position);

            if (res.fitness < population[p].best_fitness) {
                population[p].best_fitness = res.fitness;
                population[p].best_mean_error = res.mean_error;
                population[p].best_position = population[p].position;

                if (res.fitness < global_best_fitness) {
                    global_best_fitness = res.fitness;
                    global_best_mean_error = res.mean_error;
                    global_best_position = population[p].position;
                }
            }
        }

        std::println("{}Iteration: {:3}/{}{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            color::cyan, iter + 1, n_iterations, color::reset,
            global_best_fitness,
            global_best_mean_error);
    }

    PSOResult final_res;
    final_res.best_position = global_best_position;
    final_res.best_fitness = global_best_fitness;
    final_res.best_mean_error = global_best_mean_error;
    return final_res;
}