#pragma once

#include <cassert>
#include <cstddef>


struct ihcp_parameters
{
	float q{};
	float alpha{};

	bool operator==(const ihcp_parameters& other) const noexcept = default;

	float& operator[](std::size_t index) noexcept
	{
		assert(index < 2);

		return (index == 0) ? q : alpha;
	}

	constexpr float operator[](std::size_t index) const noexcept
	{
		assert(index < 2);

		return (index == 0) ? q : alpha;
	}
};