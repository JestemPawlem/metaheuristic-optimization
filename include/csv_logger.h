#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "problems/optimization_problem.h"
#include "optimization_types.h"


class csv_logger
{
public:
    using hyperparam_list = std::vector<std::pair<std::string_view, std::string>>;

    csv_logger(std::filesystem::path dir, std::string_view run_id) :
        path_(std::move(dir) / ("run_" + std::string(run_id) + ".csv"))
    {
        if (!path_.parent_path().empty())
            std::filesystem::create_directories(path_.parent_path());
    }

    template <std::size_t Dim>
    void log(
        std::string_view algorithm,
        const optimization_problem<Dim>& problem,
        const std::vector<optimization_result<Dim>>& results,
        const hyperparam_list& hyperparams) const
    {
        std::ofstream file(path_);
        if (!file.is_open())
            return;

        write_header(file, problem, hyperparams);

        for (std::size_t i{}; i < results.size(); ++i)
        {
            file << algorithm << ',' << (i + 1);

            for (std::size_t d{}; d < Dim; ++d)
                file << ',' << results[i].best_parameters[d];

            file << ',' << results[i].best_fitness;

            auto metrics = problem.compute_metrics(results[i].best_parameters);
            for (float m : metrics)
                file << ',' << m;

            file << ',' << results[i].n_evaluations;

            for (const auto& [_, value] : hyperparams)
                file << ',' << value;

            file << '\n';
        }
    }

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;

    template <std::size_t Dim>
    void write_header(
        std::ofstream& file,
        const optimization_problem<Dim>& problem,
        const hyperparam_list& hyperparams) const
    {
        file << "algorithm,run";

        for (std::string_view name : problem.parameter_names())
            file << ',' << name;

        file << ",fitness";

        for (std::string_view metric : problem.metric_names())
            file << ',' << metric;

        file << ",n_evaluations";

        for (const auto& [name, _] : hyperparams)
            file << ',' << name;

        file << '\n';
    }
};