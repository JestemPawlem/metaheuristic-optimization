#include <cstdlib>
#include <filesystem>
#include <print>
#include <span>
#include <string>

#include "csv_logger.h"
#include "ihcp_bounds.h"
#include "ihcp_model.h"
#include "optimization_result.h"
#include "optimizers/optimizer_factory.h"
#include "random.h"
#include "run_experiment.h"


int main(int argc, char* argv[])
{
    std::system("chcp 65001 > nul");

    rng::set_seed(0);

    std::filesystem::path session_dir = "results";
    std::string run_id = "0";
    std::string algorithm = "pso";
    std::size_t n_runs = 5;

    const ihcp_bounds bounds{
        { 100'000.0f, 1'000.0f },
        { 600'000.0f, 8'000.0f }
    };

    if (argc > 1) session_dir = argv[1];
    if (argc > 2) run_id = argv[2];
    if (argc > 3) algorithm = argv[3];
    if (argc > 4) n_runs = std::stoul(argv[4]);

    ihcp_model model(0.03f);

    try
    {
        std::span<char*> optimizer_args(argc > 5 ? argv + 5 : argv + argc, argc > 5 ? argc - 5 : 0);

        auto setup = optimizer_factory::create(algorithm, model, bounds, optimizer_args);

        std::println("Running {} optimization...", algorithm);
        std::vector<optimization_result> results = run_experiment(n_runs, setup.factory);

        csv_logger logger(session_dir, run_id);
        logger.log(algorithm, results, setup.hyperparams);

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