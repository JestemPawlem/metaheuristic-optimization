#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "experiment_result.h"
#include "optimizer.h"


experiment_result run_experiment(
    std::size_t n_runs,
    const std::function<std::unique_ptr<optimizer>()>& optimizer_factory);