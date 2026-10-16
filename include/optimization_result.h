#pragma once

#include <limits>

#include "ihcp_parameters.h"
#include "ihcp_result.h"


struct optimization_result
{
    ihcp_parameters best_parameters{};
    ihcp_result best_result{};

    auto operator<=>(const optimization_result& other) const noexcept
    {
        return best_result <=> other.best_result;
    }

    bool operator==(const optimization_result& other) const noexcept
    {
        return best_result == other.best_result;
    }
};