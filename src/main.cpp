#include <cstdlib>
#include <filesystem>
#include <print>
#include <span>
#include <string>
#include <vector>

#include "csv_logger.h"
#include "ihcp_model.h"
#include "optimization_types.h"
#include "optimizers/optimizer_factory.h"
#include "random.h"
#include "run_experiment.h"


int main(int argc, char* argv[])
{
    std::system("chcp 65001 > nul");
    rng::set_seed(0);

    try
    {
        std::filesystem::path session_dir = "results";
        std::string run_id = "0";

        float noise_level = 0.03f;

        std::string algorithm = "pso";
        std::size_t n_runs = 5;

        if (argc > 1) session_dir = argv[1];
        if (argc > 2) run_id = argv[2];
        if (argc > 3) noise_level = std::stof(argv[3]);
        if (argc > 4) algorithm = argv[4];
        if (argc > 5) n_runs = std::stoul(argv[5]);

        const optimization_bounds<2> bounds{
            { 100'000.0f, 1'000.0f },
            { 600'000.0f, 8'000.0f }
        };
        ihcp_model model(noise_level, bounds);

        std::span<char*> optimizer_args = (argc > 6)
            ? std::span(argv + 6, static_cast<std::size_t>(argc - 6))
            : std::span<char*>{};

        auto setup = optimizer_factory<2>::create(algorithm, model, optimizer_args);

        std::println("Running {} optimization (noise: {:.2f})...", algorithm, noise_level);
        std::vector<optimization_result<2>> results = run_experiment(n_runs, setup.factory);

        csv_logger logger(session_dir, run_id);
        logger.log(algorithm, model, results, setup.hyperparams);

        std::println("\n===== EXPERIMENT COMPLETE [{}] =====", algorithm);
        std::println("Completed runs:            {}", results.size());
        std::println("Output file:               {}", logger.path().string());
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "[ERROR] {}", e.what());
        return 1;
    }

    return 0;
}