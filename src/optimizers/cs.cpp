#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include "ihcp_model.h"
#include "optimizers/cs.h"


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

    std::random_device rd;
    std::mt19937 gen(rd());

    std::array<std::uniform_real_distribution<float>, 2> dist_pos;
    for (std::size_t d{}; d < 2; ++d)
        dist_pos[d] = std::uniform_real_distribution<float>(bounds.lower[d], bounds.upper[d]);

    std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> dist_idx(0, population_size - 1);

    constexpr float beta = 1.5f;
    const float sigma_u = std::pow(
        (std::tgamma(1.0f + beta) * std::sin(std::numbers::pi_v<float> * beta / 2.0f)) /
        (std::tgamma((1.0f + beta) / 2.0f) * beta * std::pow(2.0f, (beta - 1.0f) / 2.0f)),
        1.0f / beta
    );

    std::normal_distribution<float> dist_u(0.0f, sigma_u);
    std::normal_distribution<float> dist_v(0.0f, 1.0f);

    std::vector<nest> population(population_size);
    nest best_nest;

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
            population[i].position[d] = dist_pos[d](gen);

        population[i].phenotype = evaluate(population[i].position);

        if (population[i].phenotype < best_nest.phenotype)
            best_nest = population[i];
    }

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        for (std::size_t i{}; i < population_size; ++i)
        {
            ihcp_parameters new_pos{};

            for (std::size_t d{}; d < 2; ++d)
            {
                float u = dist_u(gen);
                float v = dist_v(gen);
                float levy_step = u / std::pow(std::abs(v), 1.0f / beta);

                float step_size = 0.3f * levy_step * (population[i].position[d] - best_nest.position[d]);

                new_pos[d] = population[i].position[d] + step_size;
                new_pos[d] = std::clamp(new_pos[d], bounds.lower[d], bounds.upper[d]);
            }

            ihcp_result result = evaluate(new_pos);

            if (result < population[i].phenotype)
            {
                population[i].position = new_pos;
                population[i].phenotype = result;

                if (result < best_nest.phenotype)
                    best_nest = population[i];
            }
        }

        // Krok 2: Odkrywanie obcych jaj (samoadaptująca mutacja różnicowa DE/rand/1)
        for (std::size_t i{}; i < population_size; ++i)
        {
            if (dist_01(gen) < pa)
            {
                std::size_t j = dist_idx(gen);
                std::size_t k = dist_idx(gen);

                while (j == i) j = dist_idx(gen);
                while (k == i || k == j) k = dist_idx(gen);

                ihcp_parameters new_pos{};
                for (std::size_t d{}; d < 2; ++d)
                {
                    new_pos[d] = population[i].position[d] + dist_01(gen) * (population[j].position[d] - population[k].position[d]);
                    new_pos[d] = std::clamp(new_pos[d], bounds.lower[d], bounds.upper[d]);
                }

                ihcp_result result = evaluate(new_pos);

                if (result < population[i].phenotype)
                {
                    population[i].position = new_pos;
                    population[i].phenotype = result;

                    if (result < best_nest.phenotype)
                        best_nest = population[i];
                }
            }
        }

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            best_nest.phenotype.fitness,
            best_nest.phenotype.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = best_nest.position;
    final_res.best_result = best_nest.phenotype;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}