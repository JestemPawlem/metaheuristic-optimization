#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <vector>

#include "ihcp_model.h"
#include "optimizers/cs.h"
#include "random.h"


cuckoo_search::cuckoo_search(
    const ihcp_model& model,
    ihcp_bounds bounds,
    std::size_t population_size,
    std::size_t n_iterations,
    float pa) :
    model(model),
    bounds(bounds),
    population_size(population_size),
    n_iterations(n_iterations),
    pa(pa)
{}

optimization_result cuckoo_search::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    constexpr float beta = 1.5f;
    const float sigma_u = std::pow(
        (std::tgamma(1.0f + beta) * std::sin(std::numbers::pi_v<float> * beta / 2.0f)) /
        (std::tgamma((1.0f + beta) / 2.0f) * beta * std::pow(2.0f, (beta - 1.0f) / 2.0f)),
        1.0f / beta
    );

    std::vector<nest> population(population_size);
    nest best;

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
            population[i].position[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

        population[i].result = evaluate(population[i].position);

        if (population[i].result < best.result)
            best = population[i];
    }

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        for (std::size_t i{}; i < population_size; ++i)
        {
            if (population[i].position == best.position)
                continue;

            ihcp_parameters new_pos{};

            for (std::size_t d{}; d < 2; ++d)
            {
                float u = rng::normal(0.0f, sigma_u);
                float abs_v = std::max(std::abs(rng::normal()), std::numeric_limits<float>::min());
                float levy_step = u / std::pow(abs_v, 1.0f / beta);

                float step_size = 0.3f * levy_step * (population[i].position[d] - best.position[d]);

                new_pos[d] = population[i].position[d] + step_size;
                new_pos[d] = std::clamp(new_pos[d], bounds.lower[d], bounds.upper[d]);
            }

            ihcp_result result = evaluate(new_pos);

            if (result < population[i].result)
            {
                population[i].position = new_pos;
                population[i].result = result;

                if (result < best.result)
                    best = population[i];
            }
        }

        for (std::size_t i{}; i < population_size; ++i)
        {
            if (rng::bernoulli(pa))
            {
                std::size_t j = rng::index(population_size);
                std::size_t k = rng::index(population_size);

                while (j == i) j = rng::index(population_size);
                while (k == i || k == j) k = rng::index(population_size);

                ihcp_parameters new_pos{};
                for (std::size_t d{}; d < 2; ++d)
                {
                    new_pos[d] = population[i].position[d] + rng::uniform() * (population[j].position[d] - population[k].position[d]);
                    new_pos[d] = std::clamp(new_pos[d], bounds.lower[d], bounds.upper[d]);
                }

                ihcp_result result = evaluate(new_pos);

                if (result < population[i].result)
                {
                    population[i].position = new_pos;
                    population[i].result = result;

                    if (result < best.result)
                        best = population[i];
                }
            }
        }

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            best.result.fitness,
            best.result.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = best.position;
    final_res.best_result = best.result;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}