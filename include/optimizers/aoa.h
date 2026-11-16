#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_types.h"
#include "optimizer.h"
#include "problems/optimization_problem.h"
#include "random.h"

template <std::size_t Dim>
class archimedes_optimization_algorithm : public optimizer<Dim>
{
public:
	using iteration_callback = typename optimizer<Dim>::iteration_callback;

	struct parameters
	{
		std::size_t n_solutions = 30;
		std::size_t n_iterations = 500;
		float c1 = 2.0f;
		float c2 = 6.0f;
		float c3 = 2.0f;
		float c4 = 0.5f;
	};

	archimedes_optimization_algorithm(
		const optimization_problem<Dim>& problem,
		const parameters& parameters) :
		problem_(problem),
		parameters_(parameters)
	{}

	optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
	{
		evaluation_counter<Dim> evaluate(problem_);
		const optimization_bounds<Dim>& bounds = problem_.bounds();
		const auto& [n_solutions, n_iterations, c1, c2, c3, c4] = parameters_;

		std::vector<solution> solutions(n_solutions);
		solution best{};

		for (std::size_t s{}; s < n_solutions; ++s)
		{
			for (std::size_t d{}; d < Dim; ++d)
			{
				solutions[s].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);
				solutions[s].acceleration[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);
			}
			solutions[s].density = rng::uniform();
			solutions[s].volume = rng::uniform();

			float fitness = evaluate(solutions[s].parameters);
			solutions[s].fitness = fitness;

			if (s == 0 || fitness < best.fitness)
				best = solutions[s];
		}

		const auto tf_operator = [n_iterations](std::size_t i)
			{
				const float progress = static_cast<float>(i + 1) / n_iterations;
				return std::exp(progress - 1.0f);
			};
		const auto d_operator = [n_iterations](std::size_t i)
			{
				const float progress = static_cast<float>(i + 1) / n_iterations;
				return std::exp(1.0f - progress) - progress;
			};

		constexpr float eps = std::numeric_limits<float>::min();
		constexpr float u = 1.0f;
		constexpr float l = 0.1f;

		for (std::size_t i{}; i < n_iterations; ++i)
		{
			const float tf = tf_operator(i);
			const float d_factor = d_operator(i);

			for (std::size_t s{}; s < n_solutions; ++s)
			{
				solutions[s].density += rng::uniform() * (best.density - solutions[s].density);
				solutions[s].volume += rng::uniform() * (best.volume - solutions[s].volume);

				const float denom = solutions[s].density * solutions[s].volume + eps;
				if (tf <= 0.5f)
				{
					const std::size_t r = rng::index(n_solutions);
					for (std::size_t d{}; d < Dim; ++d)
						solutions[s].acceleration[d] = (solutions[r].density +
							solutions[r].volume * solutions[r].acceleration[d]) / denom;
				}
				else
				{
					for (std::size_t d{}; d < Dim; ++d)
						solutions[s].acceleration[d] = (best.density +
							best.volume * best.acceleration[d]) / denom;
				}
			}

			std::array<float, Dim> min_acc = solutions[0].acceleration;
			std::array<float, Dim> max_acc = solutions[0].acceleration;
			for (std::size_t s{}; s < n_solutions; ++s)
			{
				for (std::size_t d{}; d < Dim; ++d)
				{
					min_acc[d] = std::min(min_acc[d], solutions[s].acceleration[d]);
					max_acc[d] = std::max(max_acc[d], solutions[s].acceleration[d]);
				}
			}

			for (std::size_t s{}; s < n_solutions; ++s)
			{
				if (tf <= 0.5f)
				{
					const std::size_t r = rng::index(n_solutions);
					for (std::size_t d{}; d < Dim; ++d)
					{
						const float range = max_acc[d] - min_acc[d];
						const float norm = (range > eps)
							? (solutions[s].acceleration[d] - min_acc[d]) / range
							: 0.5f;

						const float acc_norm = (u - l) * norm + l;

						solutions[s].parameters[d] +=
							c1 * rng::uniform() * acc_norm * d_factor *
							(solutions[r].parameters[d] - solutions[s].parameters[d]);
					}
				}
				else
				{
					const float t_flag = c3 * tf;
					for (std::size_t d{}; d < Dim; ++d)
					{
						const float range = max_acc[d] - min_acc[d];
						const float norm = (range > eps)
							? (solutions[s].acceleration[d] - min_acc[d]) / range
							: 0.5f;

						const float acc_norm = (u - l) * norm + l;

						const float p = 2.0f * rng::uniform() - c4;
						const float dir = (p <= 0.5f) ? 1.0f : -1.0f;

						solutions[s].parameters[d] = best.parameters[d] +
							dir * c2 * rng::uniform() * acc_norm * d_factor *
							(t_flag * best.parameters[d] - solutions[s].parameters[d]);
					}
				}

				for (std::size_t d{}; d < Dim; ++d)
				{
					solutions[s].parameters[d] = std::clamp(
						solutions[s].parameters[d],
						bounds.lower[d],
						bounds.upper[d]);
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
		std::array<float, Dim> acceleration{};
		float density{};
		float volume{};
		float fitness{};
	};
};