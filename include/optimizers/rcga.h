#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_types.h"
#include "optimizers/optimizer.h"
#include "problems/optimization_problem.h"
#include "random.h"

template <std::size_t Dim>
class real_coded_genetic_algorithm : public optimizer<Dim>
{
public:
    using iteration_callback = typename optimizer<Dim>::iteration_callback;

    real_coded_genetic_algorithm(
        const optimization_problem<Dim>& problem,
        std::size_t n_solutions,
        std::size_t n_iterations,
        float crossover_rate,
        float mutation_rate) :
        problem_(problem),
        n_solutions_(n_solutions),
        n_iterations_(n_iterations),
        crossover_rate_(crossover_rate),
        mutation_rate_(mutation_rate)
    {}

    optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
    {
        evaluation_counter<Dim> evaluate(problem_);
        const optimization_bounds<Dim>& bounds = problem_.bounds();

        std::array<float, Dim> ranges{};
        for (std::size_t d{}; d < Dim; ++d)
            ranges[d] = bounds.upper[d] - bounds.lower[d];

        std::vector<solution> solutions(n_solutions_);
        solution best;

        for (std::size_t i{}; i < n_solutions_; ++i)
        {
            for (std::size_t d{}; d < Dim; ++d)
                solutions[i].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

            solutions[i].fitness = evaluate(solutions[i].parameters);

            if (solutions[i].fitness < best.fitness)
                best = solutions[i];
        }

        std::vector<solution> next_solutions(n_solutions_);

        for (std::size_t iter{}; iter < n_iterations_; ++iter)
        {
            const float progress = static_cast<float>(iter) / n_iterations_;
            const float scale = 0.05f * std::pow(1.0f - progress, 2.0f) + 1e-5f;

            solution prev_best = best;

            for (std::size_t i{}; i < n_solutions_; i += 2)
            {
                std::size_t p1 = rng::index(n_solutions_);
                std::size_t p2 = rng::index(n_solutions_);
                const solution& parent1 = (solutions[p1].fitness < solutions[p2].fitness) ?
                    solutions[p1] : solutions[p2];

                std::size_t p3 = rng::index(n_solutions_);
                std::size_t p4 = rng::index(n_solutions_);
                const solution& parent2 = (solutions[p3].fitness < solutions[p4].fitness) ?
                    solutions[p3] : solutions[p4];

                solution offspring1, offspring2;

                if (rng::bernoulli(crossover_rate_))
                {
                    for (std::size_t d{}; d < Dim; ++d)
                    {
                        float gamma = rng::uniform();
                        offspring1.parameters[d] = gamma * parent1.parameters[d] + (1.0f - gamma) * parent2.parameters[d];
                        offspring2.parameters[d] = (1.0f - gamma) * parent1.parameters[d] + gamma * parent2.parameters[d];
                    }
                }
                else
                {
                    offspring1.parameters = parent1.parameters;
                    offspring2.parameters = parent2.parameters;
                }

                for (std::size_t d{}; d < Dim; ++d)
                {
                    float sigma = ranges[d] * scale;

                    if (rng::bernoulli(mutation_rate_))
                    {
                        offspring1.parameters[d] += rng::normal(0.0f, sigma);
                        offspring1.parameters[d] = std::clamp(offspring1.parameters[d], bounds.lower[d], bounds.upper[d]);
                    }
                    if (rng::bernoulli(mutation_rate_))
                    {
                        offspring2.parameters[d] += rng::normal(0.0f, sigma);
                        offspring2.parameters[d] = std::clamp(offspring2.parameters[d], bounds.lower[d], bounds.upper[d]);
                    }
                }

                offspring1.fitness = evaluate(offspring1.parameters);
                next_solutions[i] = offspring1;
                if (offspring1.fitness < best.fitness)
                    best = offspring1;

                if (i + 1 < n_solutions_)
                {
                    offspring2.fitness = evaluate(offspring2.parameters);
                    next_solutions[i + 1] = offspring2;
                    if (offspring2.fitness < best.fitness)
                        best = offspring2;
                }
            }

            auto worst_it = std::max_element(next_solutions.begin(), next_solutions.end(),
                [](const solution& a, const solution& b) {
                    return a.fitness < b.fitness;
                });
            *worst_it = prev_best;

            solutions = next_solutions;

            if (on_iteration)
            {
                on_iteration(iteration_data<Dim>{
                    iter + 1,
                    best.parameters,
                    best.fitness
                });
            }
        }

        return optimization_result<Dim>{
            evaluate.count(),
            best.parameters,
            best.fitness
        };
    }

private:
    const optimization_problem<Dim>& problem_;

    std::size_t n_solutions_;
    std::size_t n_iterations_;
    float crossover_rate_;
    float mutation_rate_;

    struct solution
    {
        std::array<float, Dim> parameters{};
        float fitness = std::numeric_limits<float>::max();
    };
};