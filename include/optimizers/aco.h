#pragma once

#include <cstddef>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "ihcp_parameters.h"
#include "optimizers/optimizer.h"


class ant_colony_optimization : public optimizer
{
public:
	ant_colony_optimization(
		const ihcp_model& model,
		ihcp_bounds bounds,
		std::size_t population_size,
		std::size_t n_iterations,
		float locality,
		float evaporation);

	optimization_result optimize() override;

private:
	const ihcp_model& model;
	ihcp_bounds bounds;

	std::size_t population_size;
	std::size_t n_iterations;
	float locality;
	float evaporation;

	struct ant
	{
		ihcp_parameters position{};
		ihcp_result result{};
	};
};