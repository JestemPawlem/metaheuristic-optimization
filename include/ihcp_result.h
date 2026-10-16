#pragma once

#include <limits>


struct ihcp_result
{
    float fitness = std::numeric_limits<float>::max();
    float mean_error = std::numeric_limits<float>::max();

    auto operator<=>(const ihcp_result& other) const noexcept
    {
        return fitness <=> other.fitness;
    }

    bool operator==(const ihcp_result& other) const noexcept
    {
        return fitness == other.fitness;
    }
};