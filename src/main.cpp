#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <print>
#include <string>
#include <string_view>

#include "csv_logger.h"
#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "optimization_result.h"
#include "optimizer.h"
#include "pso.h"
#include "run_experiment.h"
#include "sga.h"


int main(int argc, char* argv[])
{
    std::system("chcp 65001 > nul");

    std::filesystem::path session_dir = "results";
    std::string run_id = "0";
    std::string algorithm = "sga";
    std::size_t n_runs = 5;
    std::size_t population_size = 30;
    std::size_t n_iterations = 30;

    const ihcp_bounds bounds{
        { 100'000.0f, 1'000.0f },
        { 600'000.0f, 8'000.0f }
    };

    if (argc > 1) session_dir = argv[1];
    if (argc > 2) run_id = argv[2];
    if (argc > 3) algorithm = argv[3];
    if (argc > 4) n_runs = std::stoul(argv[4]);
    if (argc > 5) population_size = std::stoul(argv[5]);
    if (argc > 6) n_iterations = std::stoul(argv[6]);

    ihcp_model model{};
    std::function<std::unique_ptr<optimizer>()> factory;

    if (algorithm == "pso")
    {
        float inertia = 0.7f;
        float c1 = 1.5f;
        float c2 = 1.5f;

        if (argc > 7) inertia = std::stof(argv[7]);
        if (argc > 8) c1 = std::stof(argv[8]);
        if (argc > 9) c2 = std::stof(argv[9]);

        factory = [&]()
            {
                return std::make_unique<pso>(
                    model,
                    population_size,
                    n_iterations,
                    inertia,
                    c1,
                    c2,
                    bounds
                );
            };
    }
    else if (algorithm == "sga")
    {
        float crossover_rate = 0.8f;
        float mutation_rate = 0.1f;

        if (argc > 7) crossover_rate = std::stof(argv[7]);
        if (argc > 8) mutation_rate = std::stof(argv[8]);

        factory = [&]()
            {
                return std::make_unique<sga>(
                    model,
                    population_size,
                    n_iterations,
                    crossover_rate,
                    mutation_rate,
                    bounds
                );
            };
    }
    else
    {
        std::println("[ERROR] Unknown algorithm: '{}'.", algorithm);
        return 1;
    }

    std::println("Running {} optimization...", algorithm);

    std::vector<optimization_result> results = run_experiment(n_runs, factory);

    csv_logger logger(session_dir, run_id);
    logger.log(algorithm, results);

    std::println("\n===== EXPERIMENT COMPLETE [{}] =====", algorithm);
    std::println("Completed runs:            {}", results.size());
    std::println("Output file:               {}", logger.path().string());

    return 0;
}