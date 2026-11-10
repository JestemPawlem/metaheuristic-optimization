#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <print>
#include <vector>

#include "ihcp_model.h"
#include "optimizers/gwo.h"
#include "random.h"


grey_wolf_optimizer::grey_wolf_optimizer(
    const ihcp_model& model,
    ihcp_bounds bounds,
    std::size_t population_size,
    std::size_t n_iterations) :
    model(model),
    bounds(bounds),
    population_size(population_size),
    n_iterations(n_iterations)
{}


optimization_result grey_wolf_optimizer::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    std::vector<individual> population(population_size);

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
            population[i].position[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

        population[i].result = evaluate(population[i].position);
    }

    auto update_hierarchy = [](std::vector<individual>& population)
        {
            std::partial_sort(population.begin(), population.begin() + 3, population.end(),
                [](const individual& a, const individual& b)
                {
                    return a.result < b.result;
                });
        };

    update_hierarchy(population);
    individual alpha = population[0];
    individual beta = population[1];
    individual delta = population[2];

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        const float a = 2.0f * (1.0f - static_cast<float>(iter) / static_cast<float>(n_iterations));

        for (std::size_t i{}; i < population_size; ++i)
        {
            ihcp_parameters new_pos{};

            for (std::size_t d{}; d < 2; ++d)
            {
                float r1_a = rng::uniform(), r2_a = rng::uniform();
                float A1 = a * (2.0f * r1_a - 1.0f);
                float C1 = 2.0f * r2_a;
                float D_alpha = std::abs(C1 * alpha.position[d] - population[i].position[d]);
                float X1 = alpha.position[d] - A1 * D_alpha;

                float r1_b = rng::uniform(), r2_b = rng::uniform();
                float A2 = a * (2.0f * r1_b - 1.0f);
                float C2 = 2.0f * r2_b;
                float D_beta = std::abs(C2 * beta.position[d] - population[i].position[d]);
                float X2 = beta.position[d] - A2 * D_beta;

                float r1_d = rng::uniform(), r2_d = rng::uniform();
                float A3 = a * (2.0f * r1_d - 1.0f);
                float C3 = 2.0f * r2_d;
                float D_delta = std::abs(C3 * delta.position[d] - population[i].position[d]);
                float X3 = delta.position[d] - A3 * D_delta;

                new_pos[d] = (X1 + X2 + X3) / 3.0f;
                new_pos[d] = std::clamp(new_pos[d], bounds.lower[d], bounds.upper[d]);
            }

            population[i].position = new_pos;
            population[i].result = evaluate(new_pos);
        }

        update_hierarchy(population);
        alpha = population[0];
        beta = population[1];
        delta = population[2];

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            alpha.result.fitness,
            alpha.result.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = alpha.position;
    final_res.best_result = alpha.result;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}