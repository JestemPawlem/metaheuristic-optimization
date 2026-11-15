#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "optimization_types.h"


template <std::size_t Dim>
class optimization_problem
{
public:
    virtual ~optimization_problem() = default;

    static constexpr std::size_t dimension() noexcept
    {
        return Dim;
    }
    virtual const optimization_bounds<Dim>& bounds() const noexcept = 0;
    virtual const std::array<std::string_view, Dim>& parameter_names() const noexcept = 0;

    virtual float evaluate(const std::array<float, Dim>& parameters) const = 0;

    virtual std::span<const std::string_view> metric_names() const noexcept
    {
        return {};
    }

    virtual std::vector<float> compute_metrics(const std::array<float, Dim>&) const
    {
        return {};
    }
};