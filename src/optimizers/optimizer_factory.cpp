#include <cstddef>
#include <stdexcept>
#include <string>

#include "optimizers/optimizer_factory.h"
#include "optimizers/aco.h"
#include "optimizers/cs.h"
#include "optimizers/pso.h"
#include "optimizers/rcga.h"
#include "optimizers/gwo.h"


optimizer_factory::result optimizer_factory::create(
    std::string_view algorithm,
    const ihcp_model& model,
    const ihcp_bounds& bounds,
    std::span<char*> optimizer_args)
{
    if (algorithm == "aco")
    {
        std::size_t population_size = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
        std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;
        float locality = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.3f;
        float evaporation = optimizer_args.size() > 3 ? std::stof(optimizer_args[3]) : 0.85f;

        hyperparam_list params = {
            {"n_pop", std::to_string(population_size)},
            {"n_iter", std::to_string(n_iterations)},
            {"aco_locality", std::to_string(locality)},
            {"aco_evaporation", std::to_string(evaporation)}
        };

        auto factory = [&model, bounds, population_size, n_iterations, locality, evaporation]() {
            return std::make_unique<ant_colony_optimization>(
                model, bounds, population_size, n_iterations, locality, evaporation
            );
            };

        return { std::move(factory), std::move(params) };
    }
    if (algorithm == "cs")
    {
        std::size_t population_size = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
        std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 60;
        float pa = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.25f;

        hyperparam_list params = {
            {"n_pop", std::to_string(population_size)},
            {"n_iter", std::to_string(n_iterations)},
            {"cs_pa", std::to_string(pa)}
        };

        auto factory = [&model, bounds, population_size, n_iterations, pa]() {
            return std::make_unique<cuckoo_search>(
                model, bounds, population_size, n_iterations, pa
            );
            };

        return { std::move(factory), std::move(params) };
    }
    if (algorithm == "pso")
    {
        std::size_t population_size = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
        std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;
        float inertia = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.7f;
        float c1 = optimizer_args.size() > 3 ? std::stof(optimizer_args[3]) : 1.5f;
        float c2 = optimizer_args.size() > 4 ? std::stof(optimizer_args[4]) : 1.5f;

        hyperparam_list params = {
            {"n_pop", std::to_string(population_size)},
            {"n_iter", std::to_string(n_iterations)},
            {"pso_inertia", std::to_string(inertia)},
            {"pso_c1", std::to_string(c1)},
            {"pso_c2", std::to_string(c2)}
        };

        auto factory = [&model, bounds, population_size, n_iterations, inertia, c1, c2]() {
            return std::make_unique<particle_swarm_optimization>(
                model, bounds, population_size, n_iterations, inertia, c1, c2
            );
            };

        return { std::move(factory), std::move(params) };
    }

    if (algorithm == "rcga")
    {
        std::size_t population_size = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
        std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;
        float crossover_rate = optimizer_args.size() > 2 ? std::stof(optimizer_args[2]) : 0.8f;
        float mutation_rate = optimizer_args.size() > 3 ? std::stof(optimizer_args[3]) : 0.3f;

        hyperparam_list params = {
            {"n_pop", std::to_string(population_size)},
            {"n_iter", std::to_string(n_iterations)},
            {"rcga_crossover_rate", std::to_string(crossover_rate)},
            {"rcga_mutation_rate", std::to_string(mutation_rate)}
        };

        auto factory = [&model, bounds, population_size, n_iterations, crossover_rate, mutation_rate]() {
            return std::make_unique<real_coded_genetic_algorithm>(
                model, bounds, population_size, n_iterations, crossover_rate, mutation_rate
            );
            };

        return { std::move(factory), std::move(params) };
    }

    if (algorithm == "gwo")
    {
        std::size_t population_size = optimizer_args.size() > 0 ? std::stoul(optimizer_args[0]) : 20;
        std::size_t n_iterations = optimizer_args.size() > 1 ? std::stoul(optimizer_args[1]) : 80;

        hyperparam_list params = {
            {"n_pop", std::to_string(population_size)},
            {"n_iter", std::to_string(n_iterations)},
        };

        auto factory = [&model, bounds, population_size, n_iterations]() {
            return std::make_unique<grey_wolf_optimizer>(
                model, bounds, population_size, n_iterations
            );
            };

        return { std::move(factory), std::move(params) };
    }

    throw std::invalid_argument(std::string("Unknown algorithm: ") + std::string(algorithm));
}