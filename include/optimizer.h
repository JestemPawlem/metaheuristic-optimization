#pragma once

#include "ihcp_model.h"
#include "optimization_result.h"


class optimizer
{
public:
    virtual ~optimizer() = default;

    virtual optimization_result optimize() = 0;
};