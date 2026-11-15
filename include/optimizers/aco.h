#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_problem.h"
#include "optimization_types.h"
#include "optimizers/optimizer.h"
#include "random.h"


template <std::size_t Dim>
class ant_colony_optimization : public optimizer<Dim>
{
public:
    using iteration_callback = typename optimizer<Dim>::iteration_callback;

	ant_colony_optimization(
		const optimization_problem<Dim>& problem,
		std::size_t n_solutions,
		std::size_t n_iterations,
		float locality,
		float evaporation) :
		problem_(problem),
		n_solutions_(n_solutions),
		n_iterations_(n_iterations),
		locality_(locality),
		evaporation_(evaporation)
	{};

    optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
    {
        evaluation_counter<Dim> evaluate(problem_);
        const optimization_bounds<Dim>& bounds = problem_.bounds();

        std::vector<solution> solutions(n_solutions_);
        solutions.reserve(n_solutions_ * 2);

        for (std::size_t i{}; i < n_solutions_; ++i)
        {
            for (std::size_t d{}; d < Dim; ++d)
                solutions[i].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

            solutions[i].fitness = evaluate(solutions[i].parameters);
        }

        auto sort_solutions = [](std::vector<solution>& pop)
            {
                std::sort(pop.begin(), pop.end(),
                    [](const solution& a, const solution& b) {
                        return a.fitness < b.fitness;
                    });
            };

        sort_solutions(solutions);

        std::vector<float> weights(n_solutions_);
        const float denom = locality_ * n_solutions_ * std::sqrt(2.0f * std::numbers::pi_v<float>);
        for (std::size_t p{}; p < n_solutions_; ++p)
        {
            const float p_f = static_cast<float>(p);
            float exp_arg = -(p_f * p_f) / (2.0f * locality_ * locality_ * n_solutions_ * n_solutions_);
            weights[p] = (1.0f / denom) * std::exp(exp_arg);
        }

        std::vector<solution> new_solutions(n_solutions_);

        for (std::size_t i{}; i < n_iterations_; ++i)
        {
            for (std::size_t s{}; s < n_solutions_; ++s)
            {
                std::size_t l = rng::discrete(weights);

                for (std::size_t d{}; d < Dim; ++d)
                {
                    float sum_dist{};
                    for (std::size_t e{}; e < n_solutions_; ++e)
                        sum_dist += std::abs(solutions[e].parameters[d] - solutions[l].parameters[d]);

                    float sigma = evaporation_ * (sum_dist / static_cast<float>(n_solutions_ - 1));
                    sigma = std::max(sigma, 1e-6f);

                    new_solutions[s].parameters[d] = std::clamp(rng::normal(solutions[l].parameters[d], sigma),
                        bounds.lower[d], bounds.upper[d]);
                }

                new_solutions[s].fitness = evaluate(new_solutions[s].parameters);
            }

            solutions.insert(solutions.end(), new_solutions.begin(), new_solutions.end());
            sort_solutions(solutions);
            solutions.resize(n_solutions_);

            if (on_iteration)
                on_iteration(iteration_data<Dim>
                {
                    i + 1,
                    solutions.front().parameters,
                    solutions.front().fitness
                });
        }

        return optimization_result<Dim>
        {
            evaluate.count(),
            solutions.front().parameters,
            solutions.front().fitness
        };
    }

private:
	const optimization_problem<Dim>& problem_;

	std::size_t n_solutions_;
	std::size_t n_iterations_;
	float locality_;
	float evaporation_;

	struct solution
	{
		std::array<float, Dim> parameters{};
		float fitness{};
	};
};