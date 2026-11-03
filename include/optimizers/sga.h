#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "ihcp_parameters.h"
#include "optimizer.h"


class simple_genetic_algorithm : public optimizer
{
public:
    simple_genetic_algorithm(
        const ihcp_model& model,
        ihcp_bounds bounds,
        std::size_t population_size,
        std::size_t n_iterations,
        float crossover_rate,
        float mutation_rate);

    optimization_result optimize() override;

private:
    const ihcp_model& model;
    ihcp_bounds bounds;

    std::size_t population_size;
    std::size_t n_iterations;
    float crossover_rate;
    float mutation_rate;

    struct individual
    {
        ihcp_parameters genotype{};
        ihcp_result phenotype{};
    };
};