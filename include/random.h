#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rng
{
    void set_seed(std::uint32_t seed);

    float uniform();
    float uniform(float min, float max);

    std::size_t index(std::size_t size);

    float normal();
    float normal(float mean, float stddev);

    bool bernoulli(float p);

    std::size_t discrete(std::span<const float> weights);
}