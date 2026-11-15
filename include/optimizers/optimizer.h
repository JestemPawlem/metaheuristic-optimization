#pragma once

#include <cstddef>
#include <functional>

#include "optimization_types.h"


template <std::size_t Dim>
class optimizer
{
public:
    using iteration_callback = std::function<void(const iteration_data<Dim>&)>;

    virtual ~optimizer() = default;

    virtual optimization_result<Dim> optimize(iteration_callback on_iteration = {}) = 0;
};