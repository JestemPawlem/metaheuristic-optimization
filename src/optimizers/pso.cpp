#include <algorithm>
#include <limits>
#include <print>
#include <vector>

#include "ihcp_model.h"
#include "optimizers/pso.h"
#include "random.h"


particle_swarm_optimization::particle_swarm_optimization(
    const ihcp_model& model,
    ihcp_bounds bounds,
    std::size_t population_size,
    std::size_t n_iterations,
    float inertia,
    float c1,
    float c2) :
    model(model),
    bounds(bounds),
    population_size(population_size),
    n_iterations(n_iterations),
    inertia(inertia),
    c1(c1),
    c2(c2)
{}


optimization_result particle_swarm_optimization::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    ihcp_parameters ranges{};
    ihcp_parameters vmax{};

    for (std::size_t d{}; d < 2; ++d)
    {
        float range = bounds.upper[d] - bounds.lower[d];
        ranges[d] = range;
        vmax[d] = range * 0.1f;
    }

    std::vector<particle> population(population_size);
    ihcp_parameters best_position{};
    ihcp_result best_result{};

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
        {
            population[i].position[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);
            population[i].velocity[d] = rng::uniform(-ranges[d] * 0.05f, ranges[d] * 0.05f);
            population[i].best_position[d] = population[i].position[d];
        }

        ihcp_result result = evaluate(population[i].position);
        population[i].best_result = result;

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
                float r1 = rng::uniform();
                float r2 = rng::uniform();

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

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K | q: {:.4f} | alpha: {:.4f}",
            iter + 1, n_iterations,
            best_result.fitness,
            best_result.mean_error,
            best_position.q,
            best_position.alpha);
    }

    optimization_result final_res;
    final_res.best_parameters = best_position;
    final_res.best_result = best_result;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}