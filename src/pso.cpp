#include <algorithm>
#include <limits>
#include <print>
#include <random>
#include <vector>

#include "ihcp_model.h"
#include "pso.h"


particle_swarm_optimization::particle_swarm_optimization(
    const ihcp_model& model,
    std::size_t population_size,
    std::size_t n_iterations,
    float inertia,
    float c1,
    float c2,
    ihcp_bounds bounds) :
    model(model),
    population_size(population_size),
    n_iterations(n_iterations),
    inertia(inertia),
    c1(c1),
    c2(c2),
    bounds(bounds)
{}


optimization_result particle_swarm_optimization::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    std::random_device rd;
    std::mt19937 gen(rd());

    std::array<std::uniform_real_distribution<float>, 2> dist_pos;
    std::array<std::uniform_real_distribution<float>, 2> dist_vel;
    ihcp_parameters vmax{};

    for (std::size_t d{}; d < 2; ++d)
    {
        float range = bounds.upper[d] - bounds.lower[d];
        dist_pos[d] = std::uniform_real_distribution<float>(bounds.lower[d], bounds.upper[d]);
        dist_vel[d] = std::uniform_real_distribution<float>(-range * 0.05f, range * 0.05f);
        vmax[d] = range * 0.1f;
    }
    std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);

    std::vector<particle> population(population_size);
    ihcp_parameters best_position{};
    ihcp_result best_result{};

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
        {
            population[i].position[d] = dist_pos[d](gen);
            population[i].velocity[d] = dist_vel[d](gen);
            population[i].best_position[d] = population[i].position[d];
        }

        ihcp_result result = evaluate(population[i].position);

        if (result < best_result)
        {
            best_position = population[i].position;
            best_result = result;
        }
    }

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        for (std::size_t p{}; p < population_size; ++p)
        {
            for (std::size_t d{}; d < 2; ++d)
            {
                float r1 = dist_01(gen);
                float r2 = dist_01(gen);

                population[p].velocity[d] = inertia * population[p].velocity[d] +
                    c1 * r1 * (population[p].best_position[d] - population[p].position[d]) +
                    c2 * r2 * (best_position[d] - population[p].position[d]);

                population[p].velocity[d] = std::clamp(population[p].velocity[d], -vmax[d], vmax[d]);
                population[p].position[d] += population[p].velocity[d];

                if (population[p].position[d] < bounds.lower[d])
                {
                    population[p].position[d] = bounds.lower[d];
                    population[p].velocity[d] = -population[p].velocity[d];
                }
                else if (population[p].position[d] > bounds.upper[d])
                {
                    population[p].position[d] = bounds.upper[d];
                    population[p].velocity[d] = -population[p].velocity[d];
                }
            }

            ihcp_result result = evaluate(population[p].position);

            if (result < population[p].best_result)
            {
                population[p].best_position = population[p].position;
                population[p].best_result = result;

                if (result < best_result)
                {
                    best_position = population[p].position;
                    best_result = result;
                }
            }
        }

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            best_result.fitness,
            best_result.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = best_position;
    final_res.best_result = best_result;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}