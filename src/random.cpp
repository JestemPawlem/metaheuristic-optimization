#include <random>

#include "random.h"


namespace rng
{
    namespace
    {
        thread_local std::mt19937 engine{ std::random_device{}() };

        thread_local std::uniform_real_distribution<float> dist_01{ 0.0f, 1.0f };
        thread_local std::normal_distribution<float> dist_std_normal{ 0.0f, 1.0f };
    }


    void set_seed(std::uint32_t seed)
    {
        engine.seed(seed);
    }


    float uniform()
    {
        return dist_01(engine);
    }

    float uniform(float min, float max)
    {
        return min + (max - min) * dist_01(engine);
    }


    std::size_t index(std::size_t size)
    {
        std::uniform_int_distribution<std::size_t> dist{ 0, size - 1 };
        return dist(engine);
    }


    float normal()
    {
        return dist_std_normal(engine);
    }

    float normal(float mean, float stddev)
    {
        return mean + stddev * dist_std_normal(engine);
    }


    bool bernoulli(float p)
    {
        return dist_01(engine) < p;
    }


    std::size_t discrete(std::span<const float> weights)
    {
        std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        return dist(engine);
    }
}