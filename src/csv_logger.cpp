#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include "csv_logger.h"


csv_logger::csv_logger(std::filesystem::path dir, std::string_view run_id)
{
    std::filesystem::create_directories(dir);
    path_ = dir / ("run_" + std::string(run_id) + ".csv");
}

void csv_logger::write_header(const hyperparam_list& hyperparams) const
{
    std::ofstream file(path_, std::ios::out);
    if (!file.is_open())
        return;

    file << "algorithm,"
        << "run,"
        << "q,"
        << "alpha,"
        << "fitness,"
        << "mean_error,"
        << "n_evaluations";

    for (const auto& [name, _] : hyperparams)
    {
        file << ',' << name;
    }
    file << '\n';
}

void csv_logger::log(
    std::string_view algorithm,
    const std::vector<optimization_result>& results,
    const hyperparam_list& hyperparams) const
{
    write_header(hyperparams);

    std::ofstream file(path_, std::ios::app);
    if (!file.is_open())
        return;

    for (std::size_t i{}; i < results.size(); ++i)
    {
        file << algorithm << ','
            << (i + 1) << ','
            << results[i].best_parameters.q << ','
            << results[i].best_parameters.alpha << ','
            << results[i].best_result.fitness << ','
            << results[i].best_result.mean_error << ','
            << results[i].n_evaluations;

        for (const auto& [_, value] : hyperparams)
        {
            file << ',' << value;
        }
        file << '\n';
    }
}

const std::filesystem::path& csv_logger::path() const noexcept
{
    return path_;
}