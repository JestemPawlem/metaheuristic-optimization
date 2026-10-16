#include <cstddef>

#include "optimization_result.h"


struct experiment_result
{
    std::size_t total_runs{};
    std::size_t used_runs{};
    optimization_result best_result;
    float avg_fitness{};
    float avg_mean_error{};
};