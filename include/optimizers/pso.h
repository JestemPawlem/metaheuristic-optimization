#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "ihcp_parameters.h"
#include "optimizer.h"


class particle_swarm_optimization : public optimizer
{
public:
    particle_swarm_optimization(
        const ihcp_model& model,
        ihcp_bounds bounds,
        std::size_t population_size,
        std::size_t n_iterations,
        float inertia,
        float c1,
        float c2
    );

    optimization_result optimize() override;

private:
    const ihcp_model& model;
    ihcp_bounds bounds;

    std::size_t population_size;
    std::size_t n_iterations;

    float inertia;
    float c1;
    float c2;

    struct particle
    {
        ihcp_parameters position{};
        ihcp_parameters velocity{};
        ihcp_parameters best_position{};
        ihcp_result best_result{};
    };
};