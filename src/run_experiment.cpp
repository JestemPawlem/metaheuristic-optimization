#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "color.h"
#include "experiment_result.h"
#include "ihcp_model.h"
#include "optimizer.h"
#include "print.h"


experiment_result run_experiment(
    std::size_t n_runs,
    const std::function<std::unique_ptr<optimizer>()>& optimizer_factory)
{
    optimization_result best_result;

    std::vector<optimization_result> all_results;
    all_results.reserve(n_runs);

    println("{}Starting {} optimization runs for IHCP...{}", color::cyan, n_runs, color::reset);

    for (std::size_t run{}; run < n_runs; ++run)
    {
        println("\n{}--- Run {} / {} ---{}", color::yellow, run + 1, n_runs, color::reset);

        std::unique_ptr<optimizer> opt = optimizer_factory();
        optimization_result result = opt->optimize();
        all_results.push_back(result);

        if (result < best_result)
            best_result = result;
    }

    std::sort(all_results.begin(), all_results.end());

    std::size_t drop_count = static_cast<std::size_t>(n_runs * 0.2f);
    std::size_t used_runs = n_runs - drop_count;
    if (used_runs == 0)
        used_runs = 1;

    float sum_fitness = 0.0f;
    float sum_mean_error = 0.0f;

    for (std::size_t i{}; i < used_runs; ++i)
    {
        sum_fitness += all_results[i].best_result.fitness;
        sum_mean_error += all_results[i].best_result.mean_error;
    }

    experiment_result summary;
    summary.total_runs = n_runs;
    summary.used_runs = used_runs;
    summary.best_result = best_result;
    summary.avg_fitness = sum_fitness / static_cast<float>(used_runs);
    summary.avg_mean_error = sum_mean_error / static_cast<float>(used_runs);

    return summary;
}