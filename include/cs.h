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
        std::size_t population_size,
        std::size_t n_iterations,
        float pa,
        ihcp_bounds bounds);

    optimization_result optimize() override;

private:
    const ihcp_model& model;

    std::size_t population_size; // Liczba gniazd
    std::size_t n_iterations;
    float pa;                    // Prawdopodobieństwo porzucenia gniazda (często 0.25)

    ihcp_bounds bounds;

    struct nest
    {
        ihcp_parameters position{};
        ihcp_result phenotype{};
    };
};