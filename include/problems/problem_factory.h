#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "config.h"
#include "problems/benchmarks.h"
#include "problems/ihcp_model.h"
#include "problems/optimization_problem.h"

template <std::size_t Dim>
class problem_factory
{
public:
	static std::unique_ptr<optimization_problem<Dim>> create(
		std::string_view problem_name,
		const config& cfg)
	{
		if (problem_name == "ihcp")
		{
			if constexpr (Dim == 2)
			{
				const float noise_level = cfg.get<float>("ihcp.noise", 0.03f);
				const optimization_bounds<2> bounds{
					{ 100'000.0f, 1'000.0f },
					{ 600'000.0f, 8'000.0f }
				};
				return std::make_unique<ihcp_model>(noise_level, bounds);
			}
			else
			{
				throw std::invalid_argument("IHCP supports only 2 dimensions");
			}
		}

		if (problem_name == "ackley")
			return std::make_unique<ackley_problem<Dim>>();
		if (problem_name == "sphere")
			return std::make_unique<sphere_problem<Dim>>();
		if (problem_name == "rastrigin")
			return std::make_unique<rastrigin_problem<Dim>>();
		if (problem_name == "rosenbrock")
			return std::make_unique<rosenbrock_problem<Dim>>();
		if (problem_name == "bukin")
			return std::make_unique<bukin6_problem<Dim>>();
		if (problem_name == "beale")
			return std::make_unique<beale_problem<Dim>>();

		throw std::invalid_argument(std::string("Unknown problem: ") + std::string(problem_name));
	}
};