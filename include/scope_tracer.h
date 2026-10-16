#pragma once

#include "color.h"
#include "print.h"


class scope_tracer
{
public:
    explicit scope_tracer(const char* function_name)
        : name(function_name)
    {
        println("{}--> Entering: {}{}", color::red, name, color::reset);
    }

    ~scope_tracer()
    {
        println("{}<-- Exiting:  {}{}", color::red, name, color::reset);
    }

private:
    const char* name;
};

#define SCOPE_TRACER_CONCAT_IMPL(x, y) x##y
#define SCOPE_TRACER_CONCAT(x, y) SCOPE_TRACER_CONCAT_IMPL(x, y)

#define SCOPE_TRACE() scope_tracer SCOPE_TRACER_CONCAT(tracer_, __LINE__)(__FUNCTION__)