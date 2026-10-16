#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "ihcp_parameters.h"
#include "optimizer.h"


class sga : public optimizer
{
public:
    sga(
        const ihcp_model& model,
        std::size_t population_size,
        std::size_t n_iterations,
        float crossover_rate,
        float mutation_rate,
        ihcp_bounds bounds);

    optimization_result optimize() override;

private:
    const ihcp_model& model;

    std::size_t population_size;
    std::size_t n_iterations;
    float crossover_rate;
    float mutation_rate;

    ihcp_bounds bounds;

    struct individual
    {
        ihcp_parameters genotype{};
        ihcp_result phenotype{};
    };
};