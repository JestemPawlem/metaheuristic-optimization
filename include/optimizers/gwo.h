#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_problem.h"
#include "optimization_types.h"
#include "optimizers/optimizer.h"
#include "random.h"


template <std::size_t Dim>
class grey_wolf_optimizer : public optimizer<Dim>
{
public:
    using iteration_callback = typename optimizer<Dim>::iteration_callback;

    grey_wolf_optimizer(
        const optimization_problem<Dim>& problem,
        std::size_t n_solutions,
        std::size_t n_iterations) :
        problem_(problem),
        n_solutions_(n_solutions),
        n_iterations_(n_iterations)
    {}

    optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
    {
        evaluation_counter<Dim> evaluate(problem_);
        const optimization_bounds<Dim>& bounds = problem_.bounds();

        std::vector<solution> solutions(n_solutions_);

        for (std::size_t i{}; i < n_solutions_; ++i)
        {
            for (std::size_t d{}; d < Dim; ++d)
                solutions[i].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);

            solutions[i].fitness = evaluate(solutions[i].parameters);
        }

        auto update_hierarchy = [](std::vector<solution>& pop)
            {
                std::partial_sort(pop.begin(), pop.begin() + 3, pop.end(),
                    [](const solution& a, const solution& b)
                    {
                        return a.fitness < b.fitness;
                    });
            };

        update_hierarchy(solutions);
        solution alpha = solutions[0];
        solution beta = solutions[1];
        solution delta = solutions[2];

        for (std::size_t iter{}; iter < n_iterations_; ++iter)
        {
            const float a = 2.0f * (1.0f - static_cast<float>(iter) / static_cast<float>(n_iterations_));

            for (std::size_t i{}; i < n_solutions_; ++i)
            {
                std::array<float, Dim> new_parameters{};

                for (std::size_t d{}; d < Dim; ++d)
                {
                    float r1_a = rng::uniform(), r2_a = rng::uniform();
                    float A1 = a * (2.0f * r1_a - 1.0f);
                    float C1 = 2.0f * r2_a;
                    float D_alpha = std::abs(C1 * alpha.parameters[d] - solutions[i].parameters[d]);
                    float X1 = alpha.parameters[d] - A1 * D_alpha;

                    float r1_b = rng::uniform(), r2_b = rng::uniform();
                    float A2 = a * (2.0f * r1_b - 1.0f);
                    float C2 = 2.0f * r2_b;
                    float D_beta = std::abs(C2 * beta.parameters[d] - solutions[i].parameters[d]);
                    float X2 = beta.parameters[d] - A2 * D_beta;

                    float r1_d = rng::uniform(), r2_d = rng::uniform();
                    float A3 = a * (2.0f * r1_d - 1.0f);
                    float C3 = 2.0f * r2_d;
                    float D_delta = std::abs(C3 * delta.parameters[d] - solutions[i].parameters[d]);
                    float X3 = delta.parameters[d] - A3 * D_delta;

                    new_parameters[d] = std::clamp(
                        (X1 + X2 + X3) / 3.0f,
                        bounds.lower[d],
                        bounds.upper[d]
                    );
                }

                solutions[i].parameters = new_parameters;
                solutions[i].fitness = evaluate(new_parameters);
            }

            update_hierarchy(solutions);
            alpha = solutions[0];
            beta = solutions[1];
            delta = solutions[2];

            if (on_iteration)
            {
                on_iteration(iteration_data<Dim>{
                    iter + 1,
                    alpha.parameters,
                    alpha.fitness
                });
            }
        }

        return optimization_result<Dim>{
            evaluate.count(),
            alpha.parameters,
            alpha.fitness
        };
    }

private:
    const optimization_problem<Dim>& problem_;

    std::size_t n_solutions_;
    std::size_t n_iterations_;

    struct solution
    {
        std::array<float, Dim> parameters{};
        float fitness{};
    };
};