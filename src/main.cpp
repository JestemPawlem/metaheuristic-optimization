#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "config.h"
#include "csv_logger.h"
#include "optimization_types.h"
#include "optimizers/optimizer_factory.h"
#include "problems/problem_factory.h"
#include "random.h"
#include "run_experiment.h"


int main(int argc, char* argv[])
{
    try
    {
        const std::filesystem::path config_path = (argc > 1) ? argv[1] : "experiment.cfg";
        const config cfg = config::load(config_path);

        rng::set_seed(cfg.get("seed", 0));

        const std::filesystem::path session_dir = cfg.get_string("session_dir", "results");
        const std::string run_id = cfg.get_string("run_id", "0");
        const std::string problem_name = cfg.get_string("problem", "sphere3");
        const std::string algorithm = cfg.get_string("algorithm", "sca");
        const std::size_t n_runs = cfg.get("n_runs", 10uz);
        const bool verbose = cfg.get("verbose", true);

        std::string_view base_name = problem_name;
        std::size_t dim = 2;

        const std::size_t pos = problem_name.find_first_of("0123456789");
        if (pos != std::string_view::npos)
        {
            base_name = std::string_view(problem_name).substr(0, pos);
            dim = std::stoul(problem_name.substr(pos));
        }

        auto run = [&]<std::size_t Dim>()
        {
            auto model = problem_factory<Dim>::create(base_name, cfg);
            auto setup = optimizer_factory<Dim>::create(algorithm, *model, cfg);

            std::println("Running {} optimization on {}...", algorithm, problem_name);
            std::vector<optimization_result<Dim>> results =
                run_experiment(n_runs, setup.factory, verbose);

            csv_logger logger(session_dir, run_id);
            logger.log(algorithm, *model, results, setup.hyperparams);

            std::println("\n===== EXPERIMENT COMPLETE [{} | {}] =====", algorithm, problem_name);
            std::println("Completed runs:            {}", results.size());
            std::println("Output file:               {}", logger.path().string());
        };

        if (dim == 2)       run.operator()<2>();
        else if (dim == 3)  run.operator()<3>();
        else if (dim == 4)  run.operator()<4>();
        else if (dim == 5)  run.operator()<5>();
        else if (dim == 6)  run.operator()<6>();
        else if (dim == 7)  run.operator()<7>();
        else if (dim == 8)  run.operator()<8>();
        else if (dim == 9)  run.operator()<9>();
        else if (dim == 10) run.operator()<10>();
        else throw std::invalid_argument("Unsupported dimension: " + std::to_string(dim));
    }
    catch (const std::exception& e)
    {
        std::println(stderr, "[ERROR] {}", e.what());
        return 1;
    }

    return 0;
}