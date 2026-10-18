#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include "csv_logger.h"
#include "optimization_result.h"


csv_logger::csv_logger(
    std::filesystem::path dir,
    std::string_view run_id)
{
    std::filesystem::create_directories(dir);
    path_ = dir / ("run_" + std::string(run_id) + ".csv");
    ensure_header();
}


void csv_logger::log(std::string_view algorithm, const std::vector<optimization_result>& results) const
{
    std::ofstream file(path_, std::ios::app);
    if (!file.is_open())
        return;

    for (std::size_t i = 0; i < results.size(); ++i)
    {
        file << algorithm << ','
             << (i + 1) << ','
             << results[i].best_parameters.q << ','
             << results[i].best_parameters.alpha << ','
             << results[i].best_result.fitness << ','
             << results[i].best_result.mean_error << ','
             << results[i].n_evaluations << '\n';
    }
}


const std::filesystem::path& csv_logger::path() const noexcept
{
    return path_;
}


void csv_logger::ensure_header()
{
    if (std::filesystem::exists(path_) && std::filesystem::file_size(path_) > 0)
        return;

    std::ofstream file(path_, std::ios::out);
    if (!file.is_open())
        return;

    file << "algorithm,"
         << "run,"
         << "q,"
         << "alpha,"
         << "fitness,"
         << "mean_error,"
         << "n_evaluations\n";
}