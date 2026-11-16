#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "evaluation_counter.h"
#include "optimization_types.h"
#include "optimizer.h"
#include "problems/optimization_problem.h"
#include "random.h"


template <std::size_t Dim>
class particle_swarm_optimization : public optimizer<Dim>
{
public:
    using iteration_callback = typename optimizer<Dim>::iteration_callback;

    particle_swarm_optimization(
        const optimization_problem<Dim>& problem,
        std::size_t n_solutions,
        std::size_t n_iterations,
        float inertia,
        float c1,
        float c2) :
        problem_(problem),
        n_solutions_(n_solutions),
        n_iterations_(n_iterations),
        inertia_(inertia),
        c1_(c1),
        c2_(c2)
    {}

    optimization_result<Dim> optimize(iteration_callback on_iteration = {}) override
    {
        evaluation_counter<Dim> evaluate(problem_);
        const optimization_bounds<Dim>& bounds = problem_.bounds();

        std::array<float, Dim> ranges{};
        std::array<float, Dim> max_step{};
        for (std::size_t d{}; d < Dim; ++d)
        {
            ranges[d] = bounds.upper[d] - bounds.lower[d];
            max_step[d] = ranges[d] * 0.1f;
        }

        std::vector<solution> solutions(n_solutions_);
        std::array<float, Dim> best_parameters{};
        float best_fitness = std::numeric_limits<float>::max();

        for (std::size_t i{}; i < n_solutions_; ++i)
        {
            for (std::size_t d{}; d < Dim; ++d)
            {
                solutions[i].parameters[d] = rng::uniform(bounds.lower[d], bounds.upper[d]);
                solutions[i].step[d] = rng::uniform(-ranges[d] * 0.05f, ranges[d] * 0.05f);
                solutions[i].best_parameters[d] = solutions[i].parameters[d];
            }

            float fitness = evaluate(solutions[i].parameters);
            solutions[i].best_fitness = fitness;

            if (fitness < best_fitness)
            {
                best_parameters = solutions[i].parameters;
                best_fitness = fitness;
            }
        }

        for (std::size_t i{}; i < n_iterations_; ++i)
        {
            for (std::size_t s{}; s < n_solutions_; ++s)
            {
                for (std::size_t d{}; d < Dim; ++d)
                {
                    float r1 = rng::uniform();
                    float r2 = rng::uniform();

                    solutions[s].step[d] = inertia_ * solutions[s].step[d] +
                        c1_ * r1 * (solutions[s].best_parameters[d] - solutions[s].parameters[d]) +
                        c2_ * r2 * (best_parameters[d] - solutions[s].parameters[d]);

                    solutions[s].step[d] = std::clamp(solutions[s].step[d], -max_step[d], max_step[d]);
                    solutions[s].parameters[d] += solutions[s].step[d];

                    if (solutions[s].parameters[d] < bounds.lower[d])
                    {
                        solutions[s].parameters[d] = bounds.lower[d];
                        solutions[s].step[d] = -solutions[s].step[d];
                    }
                    else if (solutions[s].parameters[d] > bounds.upper[d])
                    {
                        solutions[s].parameters[d] = bounds.upper[d];
                        solutions[s].step[d] = -solutions[s].step[d];
                    }
                }

                float fitness = evaluate(solutions[s].parameters);

                if (fitness < solutions[s].best_fitness)
                {
                    solutions[s].best_parameters = solutions[s].parameters;
                    solutions[s].best_fitness = fitness;

                    if (fitness < best_fitness)
                    {
                        best_parameters = solutions[s].parameters;
                        best_fitness = fitness;
                    }
                }
            }

            if (on_iteration)
                on_iteration(iteration_data<Dim>
                {
                    i + 1,
                    best_parameters,
                    best_fitness
                });
        }

        return optimization_result<Dim>
        {
            evaluate.count(),
            best_parameters,
            best_fitness
        };
    }

private:
    const optimization_problem<Dim>& problem_;

    std::size_t n_solutions_;
    std::size_t n_iterations_;
    float inertia_;
    float c1_;
    float c2_;

    struct solution
    {
        std::array<float, Dim> parameters{};
        std::array<float, Dim> step{};
        std::array<float, Dim> best_parameters{};
        float best_fitness{};
    };
};