#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.h"
#include "optimizers/aco.h"
#include "optimizers/aoa.h"
#include "optimizers/cs.h"
#include "optimizers/gwo.h"
#include "optimizers/optimizer.h"
#include "optimizers/pso.h"
#include "optimizers/rcga.h"
#include "problems/optimization_problem.h"

template <std::size_t Dim>
class optimizer_factory
{
public:
    using factory_fn = std::function<std::unique_ptr<optimizer<Dim>>()>;
    using hyperparam_list = std::vector<std::pair<std::string_view, std::string>>;

    struct result
    {
        factory_fn factory;
        hyperparam_list hyperparams;
    };

    static result create(
        std::string_view algorithm,
        const optimization_problem<Dim>& problem,
        const config& cfg)
    {
        const std::size_t n_solutions = cfg.get<std::size_t>("n_solutions", 20);
        const std::size_t n_iterations = cfg.get<std::size_t>("n_iterations", 80);

        if (algorithm == "aco")
        {
            const float locality = cfg.get<float>("aco.locality", 0.3f);
            const float evaporation = cfg.get<float>("aco.evaporation", 0.85f);

            hyperparam_list params = {
                {"n_solutions", std::to_string(n_solutions)},
                {"n_iterations", std::to_string(n_iterations)},
                {"aco_locality", std::to_string(locality)},
                {"aco_evaporation", std::to_string(evaporation)}
            };

            auto factory = [&problem, n_solutions, n_iterations, locality, evaporation]() {
                return std::make_unique<ant_colony_optimization<Dim>>(
                    problem, n_solutions, n_iterations, locality, evaporation
                );
                };

            return { std::move(factory), std::move(params) };
        }

        if (algorithm == "aoa")
        {
            using aoa = archimedes_optimization_algorithm<Dim>;
            typename aoa::parameters p{
                .n_solutions = n_solutions,
                .n_iterations = n_iterations,
                .c1 = cfg.get<float>("aoa.c1", 2.0f),
                .c2 = cfg.get<float>("aoa.c2", 6.0f),
                .c3 = cfg.get<float>("aoa.c3", 2.0f),
                .c4 = cfg.get<float>("aoa.c4", 0.5f)
            };

            hyperparam_list params = {
                {"n_solutions", std::to_string(p.n_solutions)},
                {"n_iterations", std::to_string(p.n_iterations)},
                {"aoa_c1", std::to_string(p.c1)},
                {"aoa_c2", std::to_string(p.c2)},
                {"aoa_c3", std::to_string(p.c3)},
                {"aoa_c4", std::to_string(p.c4)}
            };

            auto factory = [&problem, p]() {
                return std::make_unique<aoa>(problem, p);
                };

            return { std::move(factory), std::move(params) };
        }

        if (algorithm == "cs")
        {
            const std::size_t cs_iterations = cfg.get<std::size_t>("n_iterations", 60);
            const float pa = cfg.get<float>("cs.pa", 0.25f);

            hyperparam_list params = {
                {"n_solutions", std::to_string(n_solutions)},
                {"n_iterations", std::to_string(cs_iterations)},
                {"cs_pa", std::to_string(pa)}
            };

            auto factory = [&problem, n_solutions, cs_iterations, pa]() {
                return std::make_unique<cuckoo_search<Dim>>(
                    problem, n_solutions, cs_iterations, pa
                );
                };

            return { std::move(factory), std::move(params) };
        }

        if (algorithm == "pso")
        {
            const float inertia = cfg.get<float>("pso.inertia", 0.7f);
            const float c1 = cfg.get<float>("pso.c1", 1.5f);
            const float c2 = cfg.get<float>("pso.c2", 1.5f);

            hyperparam_list params = {
                {"n_solutions", std::to_string(n_solutions)},
                {"n_iterations", std::to_string(n_iterations)},
                {"pso_inertia", std::to_string(inertia)},
                {"pso_c1", std::to_string(c1)},
                {"pso_c2", std::to_string(c2)}
            };

            auto factory = [&problem, n_solutions, n_iterations, inertia, c1, c2]() {
                return std::make_unique<particle_swarm_optimization<Dim>>(
                    problem, n_solutions, n_iterations, inertia, c1, c2
                );
                };

            return { std::move(factory), std::move(params) };
        }

        if (algorithm == "rcga")
        {
            const float crossover_rate = cfg.get<float>("rcga.crossover_rate", 0.8f);
            const float mutation_rate = cfg.get<float>("rcga.mutation_rate", 0.3f);

            hyperparam_list params = {
                {"n_solutions", std::to_string(n_solutions)},
                {"n_iterations", std::to_string(n_iterations)},
                {"rcga_crossover_rate", std::to_string(crossover_rate)},
                {"rcga_mutation_rate", std::to_string(mutation_rate)}
            };

            auto factory = [&problem, n_solutions, n_iterations, crossover_rate, mutation_rate]() {
                return std::make_unique<real_coded_genetic_algorithm<Dim>>(
                    problem, n_solutions, n_iterations, crossover_rate, mutation_rate
                );
                };

            return { std::move(factory), std::move(params) };
        }

        if (algorithm == "gwo")
        {
            hyperparam_list params = {
                {"n_solutions", std::to_string(n_solutions)},
                {"n_iterations", std::to_string(n_iterations)}
            };

            auto factory = [&problem, n_solutions, n_iterations]() {
                return std::make_unique<grey_wolf_optimizer<Dim>>(
                    problem, n_solutions, n_iterations
                );
                };

            return { std::move(factory), std::move(params) };
        }

        throw std::invalid_argument(std::string("Unknown algorithm: ") + std::string(algorithm));
    }
};