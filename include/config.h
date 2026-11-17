#pragma once

#include <charconv>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

class config
{
public:
	static config load(const std::filesystem::path& file_path)
	{
		if (!std::filesystem::exists(file_path))
			return {};

		std::ifstream file(file_path);
		if (!file)
			throw std::runtime_error("Cannot open config file: " + file_path.string());

		config cfg;
		std::string line;
		std::size_t line_num{};

		while (std::getline(file, line))
		{
			++line_num;

			if (const auto comment_pos = line.find('#'); comment_pos != std::string::npos)
				line.erase(comment_pos);

			const std::string_view view = trim(line);
			if (view.empty())
				continue;

			const auto eq_pos = view.find('=');
			if (eq_pos == std::string_view::npos)
				throw std::runtime_error("Syntax error in config at line " + std::to_string(line_num));

			const std::string_view key = trim(view.substr(0, eq_pos));
			const std::string_view val = trim(view.substr(eq_pos + 1));

			if (key.empty())
				throw std::runtime_error("Empty key in config at line " + std::to_string(line_num));

			cfg.values_[std::string(key)] = std::string(val);
		}

		return cfg;
	}

	std::string get_string(std::string key, std::string default_val) const
	{
		if (const auto it = values_.find(key); it != values_.end())
			return it->second;
		return default_val;
	}

	template <typename T>
		requires std::integral<T> || std::floating_point<T>
	T get(std::string key, T default_val) const
	{
		const auto it = values_.find(key);
		if (it == values_.end())
			return default_val;

		const std::string_view sv = it->second;

		if constexpr (std::same_as<T, bool>)
		{
			if (sv == "true")
				return true;
			if (sv == "false")
				return false;

			throw std::runtime_error("Invalid boolean value for key '" + key + "': " + it->second);
		}
		else
		{
			T result{};
			const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);

			if (ec != std::errc{} || ptr != sv.data() + sv.size())
				throw std::runtime_error("Invalid numeric value for key '" + key + "': " + it->second);

			return result;
		}
	}

private:
	static constexpr std::string_view trim(std::string_view s) noexcept
	{
		constexpr std::string_view whitespace = " \t\r\n";
		const auto start = s.find_first_not_of(whitespace);
		if (start == std::string_view::npos)
			return {};
		const auto end = s.find_last_not_of(whitespace);
		return s.substr(start, end - start + 1);
	}

	std::unordered_map<std::string, std::string> values_;
};