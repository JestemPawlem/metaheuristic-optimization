#include <algorithm>
#include <limits>
#include <print>
#include <vector>

#include "ihcp_model.h"
#include "optimizers/rcga.h"
#include "random.h"

real_coded_genetic_algorithm::real_coded_genetic_algorithm(
    const ihcp_model& model,
    ihcp_bounds bounds,
    std::size_t population_size,
    std::size_t n_iterations,
    float crossover_rate,
    float mutation_rate) :
    model(model),
    bounds(bounds),
    population_size(population_size),
    n_iterations(n_iterations),
    crossover_rate(crossover_rate),
    mutation_rate(mutation_rate)
{}

optimization_result real_coded_genetic_algorithm::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    std::array<float, 2> ranges{};

    for (std::size_t d{}; d < 2; ++d)
        ranges[d] = bounds.upper[d] - bounds.lower[d];

    std::vector<individual> population(population_size);
    individual best;

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
            population[i].genotype[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

        population[i].phenotype = evaluate(population[i].genotype);

        if (i == 0 || population[i].phenotype < best.phenotype)
            best = population[i];
    }

    std::vector<individual> next_population(population_size);

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        individual prev_best = best;

        for (std::size_t i{ 0 }; i < population_size; i += 2)
        {
            std::size_t p1 = rng::index(population_size);
            std::size_t p2 = rng::index(population_size);
            const individual& parent1 = (population[p1].phenotype < population[p2].phenotype) ?
                population[p1] : population[p2];

            std::size_t p3 = rng::index(population_size);
            std::size_t p4 = rng::index(population_size);
            const individual& parent2 = (population[p3].phenotype < population[p4].phenotype) ?
                population[p3] : population[p4];

            individual offspring1, offspring2;

            if (rng::bernoulli(crossover_rate))
            {
                for (std::size_t d{}; d < 2; ++d)
                {
                    float gamma = rng::uniform();
                    offspring1.genotype[d] = gamma * parent1.genotype[d] + (1.0f - gamma) * parent2.genotype[d];
                    offspring2.genotype[d] = (1.0f - gamma) * parent1.genotype[d] + gamma * parent2.genotype[d];
                }
            }
            else
            {
                offspring1.genotype = parent1.genotype;
                offspring2.genotype = parent2.genotype;
            }

            for (std::size_t d{}; d < 2; ++d)
            {
                float sigma = ranges[d] * 0.05f;

                if (rng::bernoulli(mutation_rate))
                {
                    offspring1.genotype[d] += rng::normal(0.0f, sigma);
                    offspring1.genotype[d] = std::clamp(offspring1.genotype[d], bounds.lower[d], bounds.upper[d]);
                }
                if (rng::bernoulli(mutation_rate))
                {
                    offspring2.genotype[d] += rng::normal(0.0f, sigma);
                    offspring2.genotype[d] = std::clamp(offspring2.genotype[d], bounds.lower[d], bounds.upper[d]);
                }
            }

            offspring1.phenotype = evaluate(offspring1.genotype);
            next_population[i] = offspring1;
            if (offspring1.phenotype < best.phenotype)
                best = offspring1;

            if (i + 1 < population_size)
            {
                offspring2.phenotype = evaluate(offspring2.genotype);
                next_population[i + 1] = offspring2;
                if (offspring2.phenotype < best.phenotype)
                    best = offspring2;
            }
        }

        auto worst_it = std::max_element(next_population.begin(), next_population.end(),
            [](const individual& a, const individual& b) {
                return a.phenotype < b.phenotype;
            });
        *worst_it = prev_best;

        population = next_population;

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            best.phenotype.fitness,
            best.phenotype.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = best.genotype;
    final_res.best_result = best.phenotype;
    final_res.n_evaluations = n_evaluations;
    return final_res;
}