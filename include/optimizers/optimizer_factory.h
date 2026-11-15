#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "optimization_problem.h"
#include "optimizers/aco.h"
#include "optimizers/cs.h"
#include "optimizers/gwo.h"
#include "optimizers/optimizer.h"
#include "optimizers/pso.h"
#include "optimizers/rcga.h"


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
        std::span<char*> optimizer_args)
    {
        if (algorithm == "aco")
        {
            std::size_t n_solutions = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
            std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;
            float locality = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.3f;
            float evaporation = optimizer_args.size() > 3 ? std::stof(optimizer_args[3]) : 0.85f;

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

        if (algorithm == "cs")
        {
            std::size_t n_solutions = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
            std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 60;
            float pa = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.25f;

            hyperparam_list params = {
                {"n_solutions", std::to_string(n_solutions)},
                {"n_iterations", std::to_string(n_iterations)},
                {"cs_pa", std::to_string(pa)}
            };

            auto factory = [&problem, n_solutions, n_iterations, pa]() {
                return std::make_unique<cuckoo_search<Dim>>(
                    problem, n_solutions, n_iterations, pa
                );
                };

            return { std::move(factory), std::move(params) };
        }

        if (algorithm == "pso")
        {
            std::size_t n_solutions = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
            std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;
            float inertia = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.7f;
            float c1 = optimizer_args.size() > 3 ? std::stof(optimizer_args[3]) : 1.5f;
            float c2 = optimizer_args.size() > 4 ? std::stof(optimizer_args[4]) : 1.5f;

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
            std::size_t n_solutions = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
            std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;
            float crossover_rate = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.8f;
            float mutation_rate = optimizer_args.size() > 3 ? std::stof(optimizer_args[3]) : 0.3f;

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
            std::size_t n_solutions = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
            std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;

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