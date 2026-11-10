#pragma once

#include <cstddef>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "ihcp_parameters.h"
#include "optimizers/optimizer.h"


class grey_wolf_optimizer : public optimizer
{
public:
    grey_wolf_optimizer(
        const ihcp_model& model,
        ihcp_bounds bounds,
        std::size_t population_size,
        std::size_t n_iterations);

    optimization_result optimize() override;

private:
    const ihcp_model& model;
    ihcp_bounds bounds;

    std::size_t population_size;
    std::size_t n_iterations;

    struct individual
    {
        ihcp_parameters position{};
        ihcp_result result{};
    };
};