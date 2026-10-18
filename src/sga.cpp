#include <algorithm>
#include <limits>
#include <print>
#include <random>
#include <vector>

#include "ihcp_model.h"
#include "sga.h"


sga::sga(
    const ihcp_model& model,
    std::size_t population_size,
    std::size_t n_iterations,
    float crossover_rate,
    float mutation_rate,
    ihcp_bounds bounds) :
    model(model),
    population_size(population_size),
    n_iterations(n_iterations),
    crossover_rate(crossover_rate),
    mutation_rate(mutation_rate),
    bounds(bounds)
{}

optimization_result sga::optimize()
{
    std::size_t n_evaluations{};
    auto evaluate = [&](const ihcp_parameters& params)
        {
            ++n_evaluations;
            return model.objective_function(params);
        };

    std::random_device rd;
    std::mt19937 gen(rd());

    std::array<std::uniform_real_distribution<float>, 2> dist_genes;
    std::array<float, 2> gene_range{};

    for (std::size_t d{}; d < 2; ++d)
    {
        gene_range[d] = bounds.upper[d] - bounds.lower[d];
        dist_genes[d] = std::uniform_real_distribution<float>(bounds.lower[d], bounds.upper[d]);
    }

    std::uniform_real_distribution<float> dist_01(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> dist_idx(0, population_size - 1);

    std::vector<individual> population(population_size);
    individual best_individual;

    for (std::size_t i{}; i < population_size; ++i)
    {
        for (std::size_t d{}; d < 2; ++d)
            population[i].genotype[d] = dist_genes[d](gen);

        population[i].phenotype = evaluate(population[i].genotype);

        if (population[i].phenotype < best_individual.phenotype)
            best_individual = population[i];
    }

    std::vector<individual> next_population(population_size);

    for (std::size_t iter{}; iter < n_iterations; ++iter)
    {
        next_population[0] = best_individual;

        for (std::size_t i{ 1 }; i < population_size; ++i)
        {
            std::size_t p1 = dist_idx(gen);
            std::size_t p2 = dist_idx(gen);
            const individual& parent1 = (population[p1].phenotype < population[p2].phenotype) ?
                population[p1] : population[p2];

            std::size_t p3 = dist_idx(gen);
            std::size_t p4 = dist_idx(gen);
            const individual& parent2 = (population[p3].phenotype < population[p4].phenotype) ?
                population[p3] : population[p4];

            individual offspring;

            if (dist_01(gen) < crossover_rate)
            {
                float gamma = dist_01(gen);
                for (std::size_t d{}; d < 2; ++d)
                    offspring.genotype[d] = gamma * parent1.genotype[d] + (1.0f - gamma) * parent2.genotype[d];
            }
            else
                offspring.genotype = parent1.genotype;

            for (std::size_t d{}; d < 2; ++d)
                if (dist_01(gen) < mutation_rate)
                {
                    std::normal_distribution<float> mut_dist(0.0f, gene_range[d] * 0.05f);
                    offspring.genotype[d] += mut_dist(gen);
                    offspring.genotype[d] = std::clamp(offspring.genotype[d], bounds.lower[d], bounds.upper[d]);
                }

            offspring.phenotype = evaluate(offspring.genotype);

            next_population[i] = offspring;

            if (offspring.phenotype < best_individual.phenotype)
                best_individual = offspring;
        }

        population = next_population;

        std::println("Iteration: {:3}/{}  |  Best Fitness: {:10.4f}  |  Mean Error: {:10.6f} K",
            iter + 1, n_iterations,
            best_individual.phenotype.fitness,
            best_individual.phenotype.mean_error);
    }

    optimization_result final_res;
    final_res.best_parameters = best_individual.genotype;
    final_res.best_result = best_individual.phenotype;
    return final_res;
}