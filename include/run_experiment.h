#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "optimizers/optimizer.h"


std::vector<optimization_result> run_experiment(
    std::size_t n_runs,
    const std::function<std::unique_ptr<optimizer>()>& optimizer_factory);