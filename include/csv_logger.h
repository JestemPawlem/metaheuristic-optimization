#pragma once

#include <filesystem>
#include <string_view>
#include <vector>

#include "optimization_result.h"


class csv_logger
{
public:
	explicit csv_logger(
		std::filesystem::path dir,
		std::string_view run_id);

	void log(
		std::string_view algorithm,
		const std::vector<optimization_result>& results) const;

	[[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
	std::filesystem::path path_;

	void ensure_header();
};