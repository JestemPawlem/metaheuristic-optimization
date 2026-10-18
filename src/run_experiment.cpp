#include <cstddef>
#include <functional>
#include <memory>
#include <print>
#include <vector>

#include "ihcp_model.h"
#include "optimizer.h"
#include "run_experiment.h"


std::vector<optimization_result> run_experiment(
    std::size_t n_runs,
    const std::function<std::unique_ptr<optimizer>()>& optimizer_factory)
{
    std::vector<optimization_result> results;
    results.reserve(n_runs);

    std::println("Starting {} optimization runs for IHCP...", n_runs);

    for (std::size_t run{}; run < n_runs; ++run)
    {
        std::println("\n--- Run {} / {} ---", run + 1, n_runs);

        std::unique_ptr<optimizer> opt = optimizer_factory();
        optimization_result result = opt->optimize();
        results.push_back(result);
    }

    return results;
}