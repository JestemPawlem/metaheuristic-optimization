#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "ihcp_parameters.h"
#include "optimizer.h"

class cuckoo_search : public optimizer
{
public:
    cuckoo_search(
        const ihcp_model& model,
        ihcp_bounds bounds,
        std::size_t population_size,
        std::size_t n_iterations,
        float pa);

    optimization_result optimize() override;

private:
    const ihcp_model& model;
    ihcp_bounds bounds;

    std::size_t population_size;
    std::size_t n_iterations;
    float pa;

    struct nest
    {
        ihcp_parameters position{};
        ihcp_result phenotype{};
    };
};