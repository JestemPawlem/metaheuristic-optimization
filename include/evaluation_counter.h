#pragma once

#include <array>
#include <cstddef>

#include "optimization_problem.h"


template <std::size_t Dim>
class evaluation_counter
{
public:
    explicit evaluation_counter(const optimization_problem<Dim>& problem) noexcept :
        problem(problem)
    {}

    float operator()(const std::array<float, Dim>& point)
    {
        ++count_;
        return problem.evaluate(point);
    }

    std::size_t count() const noexcept
    {
        return count_;
    }

private:
    const optimization_problem<Dim>& problem;
    std::size_t count_{};
};