#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <print>
#include <vector>

#include "ihcp_model.h"
#include "optimizers/aco.h"
#include "random.h"

ant_colony_optimization::ant_colony_optimization(
    const ihcp_model& model,
    ihcp_bounds bounds,
    std::size_t population_size,
    std::size_t n_iterations,
    float locality,
    float evaporation) :
    model(model),
    bounds(bounds),
    population_size(population_size),
    n_iterations(n_iterations),
    locality(locality),
    evaporation(evaporation)
{}

optimization_result ant_colony_optimization::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    std::vector<ant> population(population_size);
    population.reserve(population_size * 2);

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
            population[i].position[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

        population[i].result = evaluate(population[i].position);
    }

    auto sort_population = [](std::vector<ant>& pop)
        {
            std::sort(pop.begin(), pop.end(),
                [](const ant& a, const ant& b) {
                    return a.result < b.result;
                });
        };

    sort_population(population);

    std::vector<float> weights(population_size);
    const float denom = locality * population_size * std::sqrt(2.0f * std::numbers::pi_v<float>);
    for (std::size_t p{}; p < population_size; ++p)
    {
        float exp_arg = -(p * p) / (2.0f * locality * locality * population_size * population_size);
        weights[p] = (1.0f / denom) * std::exp(exp_arg);
    }

    std::vector<ant> new_ants(population_size);

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        for (std::size_t p{}; p < population_size; ++p)
        {
            std::size_t l = rng::discrete(weights);

            for (std::size_t d{}; d < 2; ++d)
            {
                float sum_dist{};
                for (std::size_t e{}; e < population_size; ++e)
                    sum_dist += std::abs(population[e].position[d] - population[l].position[d]);

                float sigma = evaporation * (sum_dist / static_cast<float>(population_size - 1));
                sigma = std::max(sigma, 1e-6f);

                new_ants[p].position[d] = std::clamp(rng::normal(population[l].position[d], sigma),
                    bounds.lower[d], bounds.upper[d]);
            }

            new_ants[p].result = evaluate(new_ants[p].position);
        }

        population.insert(population.end(), new_ants.begin(), new_ants.end());
        sort_population(population);
        population.resize(population_size);

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            population.front().result.fitness,
            population.front().result.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = population.front().position;
    final_res.best_result = population.front().result;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}