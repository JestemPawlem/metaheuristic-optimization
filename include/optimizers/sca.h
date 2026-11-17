#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_types.h"
#include "optimizer.h"
#include "problems/optimization_problem.h"
#include "random.h"


template <std::size_t Dim>
class sine_cosine_algorithm : public optimizer<Dim>
{
public:
	using iteration_callback = typename optimizer<Dim>::iteration_callback;

	struct parameters
	{
		std::size_t n_solutions{};
		std::size_t n_iterations{};
		float a = 2.0f;
	};

	sine_cosine_algorithm(
		const optimization_problem<Dim>& problem,
		const parameters& parameters) :
		problem_(problem),
		parameters_(parameters)
	{}

	optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
	{
		constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;

		evaluation_counter<Dim> evaluate(problem_);
		const optimization_bounds<Dim>& bounds = problem_.bounds();
		const auto& [n_solutions, n_iterations, a] = parameters_;

		std::vector<solution> solutions(n_solutions);
		solution best{};

		for (std::size_t s{}; s < n_solutions; ++s)
		{
			for (std::size_t d{}; d < Dim; ++d)
				solutions[s].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

			const float fitness = evaluate(solutions[s].parameters);
			solutions[s].fitness = fitness;

			if (s == 0 || fitness < best.fitness)
				best = solutions[s];
		}

		for (std::size_t i{}; i < n_iterations; ++i)
		{
			const float progress = static_cast<float>(i + 1) / n_iterations;
			const float r1 = a * (1.0f - progress);

			for (std::size_t s{}; s < n_solutions; ++s)
			{
				for (std::size_t d{}; d < Dim; ++d)
				{
					const float r2 = rng::uniform(0.0f, two_pi);
					const float r3 = rng::uniform(0.0f, 2.0f);

					const float distance = std::abs(r3 * best.parameters[d] - solutions[s].parameters[d]);
					const float trig_factor = rng::bernoulli(0.5f) ? std::sin(r2) : std::cos(r2);

					solutions[s].parameters[d] = std::clamp(
						solutions[s].parameters[d] + r1 * trig_factor * distance,
						bounds.lower[d],
						bounds.upper[d]
					);
				}

				const float fitness = evaluate(solutions[s].parameters);
				solutions[s].fitness = fitness;

				if (fitness < best.fitness)
					best = solutions[s];
			}

			if (on_iteration)
			{
				on_iteration(iteration_data<Dim>
				{
					i + 1,
						best.parameters,
						best.fitness
				});
			}
		}

		return optimization_result<Dim>
		{
			evaluate.count(),
				best.parameters,
				best.fitness
		};
	}

private:
	const optimization_problem<Dim>& problem_;
	const parameters parameters_;

	struct solution
	{
		std::array<float, Dim> parameters{};
		float fitness{};
	};
};