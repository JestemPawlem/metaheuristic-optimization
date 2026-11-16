#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_types.h"
#include "optimizers/optimizer.h"
#include "problems/optimization_problem.h"
#include "random.h"


template <std::size_t Dim>
class cuckoo_search : public optimizer<Dim>
{
public:
    using iteration_callback = typename optimizer<Dim>::iteration_callback;

    cuckoo_search(
        const optimization_problem<Dim>& problem,
        std::size_t n_solutions,
        std::size_t n_iterations,
        float pa) :
        problem_(problem),
        n_solutions_(n_solutions),
        n_iterations_(n_iterations),
        pa_(pa)
    {}

    optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
    {
        evaluation_counter<Dim> evaluate(problem_);
        const optimization_bounds<Dim>& bounds = problem_.bounds();

        constexpr float beta = 1.5f;
        const float sigma_u = std::pow(
            (std::tgamma(1.0f + beta) * std::sin(std::numbers::pi_v<float> *beta / 2.0f)) /
            (std::tgamma((1.0f + beta) / 2.0f) * beta * std::pow(2.0f, (beta - 1.0f) / 2.0f)),
            1.0f / beta
        );

        std::vector<solution> solutions(n_solutions_);
        solution best;

        for (std::size_t i{}; i < n_solutions_; ++i)
        {
            for (std::size_t d{}; d < Dim; ++d)
                solutions[i].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

            solutions[i].fitness = evaluate(solutions[i].parameters);

            if (solutions[i].fitness < best.fitness)
                best = solutions[i];
        }

        for (std::size_t iter{}; iter < n_iterations_; ++iter)
        {
            for (std::size_t i{}; i < n_solutions_; ++i)
            {
                if (solutions[i].parameters == best.parameters)
                    continue;

                std::array<float, Dim> new_parameters{};

                for (std::size_t d{}; d < Dim; ++d)
                {
                    float u = rng::normal(0.0f, sigma_u);
                    float abs_v = std::max(std::abs(rng::normal()), std::numeric_limits<float>::min());
                    float levy_step = u / std::pow(abs_v, 1.0f / beta);

                    float step_size = 0.3f * levy_step * (solutions[i].parameters[d] - best.parameters[d]);

                    new_parameters[d] = std::clamp(
                        solutions[i].parameters[d] + step_size,
                        bounds.lower[d],
                        bounds.upper[d]
                    );
                }

                float fitness = evaluate(new_parameters);

                if (fitness < solutions[i].fitness)
                {
                    solutions[i].parameters = new_parameters;
                    solutions[i].fitness = fitness;

                    if (fitness < best.fitness)
                        best = solutions[i];
                }
            }

            for (std::size_t i{}; i < n_solutions_; ++i)
            {
                if (rng::bernoulli(pa_))
                {
                    std::size_t j = rng::index(n_solutions_);
                    std::size_t k = rng::index(n_solutions_);

                    while (j == i) j = rng::index(n_solutions_);
                    while (k == i || k == j) k = rng::index(n_solutions_);

                    std::array<float, Dim> new_parameters{};
                    for (std::size_t d{}; d < Dim; ++d)
                    {
                        new_parameters[d] = solutions[i].parameters[d] +
                            rng::uniform() * (solutions[j].parameters[d] - solutions[k].parameters[d]);

                        new_parameters[d] = std::clamp(
                            new_parameters[d],
                            bounds.lower[d],
                            bounds.upper[d]
                        );
                    }

                    float fitness = evaluate(new_parameters);

                    if (fitness < solutions[i].fitness)
                    {
                        solutions[i].parameters = new_parameters;
                        solutions[i].fitness = fitness;

                        if (fitness < best.fitness)
                            best = solutions[i];
                    }
                }
            }

            if (on_iteration)
            {
                on_iteration(iteration_data<Dim>{
                    iter + 1,
                    best.parameters,
                    best.fitness
                });
            }
        }

        return optimization_result<Dim>{
            evaluate.count(),
            best.parameters,
            best.fitness
        };
    }

private:
    const optimization_problem<Dim>& problem_;

    std::size_t n_solutions_;
    std::size_t n_iterations_;
    float pa_;

    struct solution
    {
        std::array<float, Dim> parameters{};
        float fitness = std::numeric_limits<float>::max();
    };
};