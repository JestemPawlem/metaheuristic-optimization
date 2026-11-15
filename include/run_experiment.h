#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <print>
#include <vector>

#include "optimization_types.h"
#include "optimizers/optimizer.h"

template <std::size_t Dim>
std::vector<optimization_result<Dim>> run_experiment(
    std::size_t n_runs,
    const std::function<std::unique_ptr<optimizer<Dim>>()>& optimizer_factory)
{
    std::vector<optimization_result<Dim>> results;
    results.reserve(n_runs);

    auto on_iteration = [](const iteration_data<Dim>& d)
        {
            std::print("Iter: {:3} | Fitness: {:12.4f} | Params: [", d.iteration, d.best_fitness);
            for (std::size_t i{}; i < Dim; ++i)
                std::print("{}{:.4f}", (i > 0 ? ", " : ""), d.best_parameters[i]);
            std::println("]");
        };

    std::println("Starting {} optimization runs...", n_runs);

    for (std::size_t run{}; run < n_runs; ++run)
    {
        std::println("\n--- Run {} / {} ---", run + 1, n_runs);

        std::unique_ptr<optimizer<Dim>> opt = optimizer_factory();
        optimization_result<Dim> result = opt->optimize(on_iteration);
        results.push_back(result);
    }

    return results;
}