#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

#include "color.h"
#include "ihcp_model.h"
#include "run_experiment.h"
#include "pso.h"
#include "print.h"


int main(int argc, char* argv[])
{
    std::system("chcp 65001 > nul");

    std::size_t n_runs = 5;
    std::size_t population_size = 20;
    std::size_t n_iterations = 20;
    float inertia = 0.7f;
    float c1 = 1.5f;
    float c2 = 1.5f;

    ihcp_bounds bounds = {
        { 100'000.0f, 1'000.0f },
        { 600'000.0f, 8'000.0f }
    };

    if (argc > 1) n_runs = std::stoul(argv[1]);
    if (argc > 2) population_size = std::stoul(argv[2]);
    if (argc > 3) n_iterations = std::stoul(argv[3]);
    if (argc > 4) inertia = std::stof(argv[4]);
    if (argc > 5) c1 = std::stof(argv[5]);
    if (argc > 6) c2 = std::stof(argv[6]);

    ihcp_model model{};
    auto pso_factory = [&]() -> std::unique_ptr<optimizer>
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

    experiment_result summary = run_experiment(n_runs, pso_factory);

    println("\n{0}===== OPTIMIZATION SUMMARY ====={1}", color::cyan, color::reset);
    println("Total runs performed:     {}", summary.total_runs);
    println("Runs used for average:    {}", summary.used_runs);

    println("\n--- BEST OVERALL RESULT ---");
    println("Optimal heat flux (q):                 {}{:.2f}{}", color::green, summary.best_result.best_parameters.q, color::reset);
    println("Optimal heat transfer coeff. (alpha):  {}{:.2f}{}", color::green, summary.best_result.best_parameters.alpha, color::reset);
    println("Best Fitness value:                    {:.4f}", summary.best_result.best_result.fitness);
    println("Best Mean Error:                       {:.6f} K", summary.best_result.best_result.mean_error);

    println("\n--- TRIMMED MEAN VALUES ---");
    println("Average Fitness value:                 {:.4f}", summary.avg_fitness);
    println("Average Mean Error:                    {:.6f} K", summary.avg_mean_error);

    return 0;
}