#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "optimization_result.h"

class csv_logger
{
public:
    using hyperparam_list = std::vector<std::pair<std::string, std::string>>;

    csv_logger(std::filesystem::path dir, std::string_view run_id);

    void log(
        std::string_view algorithm,
        const std::vector<optimization_result>& results,
        const hyperparam_list& hyperparams) const;

    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    std::filesystem::path path_;

    void write_header(const hyperparam_list& hyperparams) const;
};