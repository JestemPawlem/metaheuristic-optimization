#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>


namespace rng
{
    namespace detail
    {
        inline thread_local std::mt19937 engine{ std::random_device{}() };
        inline thread_local std::uniform_real_distribution<float> dist_01{ 0.0f, 1.0f };
        inline thread_local std::normal_distribution<float> dist_std_normal{ 0.0f, 1.0f };
    }


    inline void set_seed(std::uint32_t seed)
    {
        detail::engine.seed(seed);
    }

    inline float uniform()
    {
        return detail::dist_01(detail::engine);
    }

    inline float uniform(float min, float max)
    {
        return min + (max - min) * detail::dist_01(detail::engine);
    }

    inline std::size_t index(std::size_t size)
    {
        std::uniform_int_distribution<std::size_t> dist{ 0, size - 1 };
        return dist(detail::engine);
    }

    inline float normal()
    {
        return detail::dist_std_normal(detail::engine);
    }

    inline float normal(float mean, float stddev)
    {
        return mean + stddev * detail::dist_std_normal(detail::engine);
    }

    inline bool bernoulli(float p)
    {
        return detail::dist_01(detail::engine) < p;
    }

    inline std::size_t discrete(std::span<const float> weights)
    {
        std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        return dist(detail::engine);
    }
}