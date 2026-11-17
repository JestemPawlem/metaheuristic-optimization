#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>

#include "optimization_problem.h"
#include "optimization_types.h"

template <std::size_t Dim>
class benchmark_problem : public optimization_problem<Dim>
{
public:
	const optimization_bounds<Dim>& bounds() const noexcept override
	{
		return bounds_;
	}

	const std::array<std::string, Dim>& parameter_names() const noexcept override
	{
		return parameter_names_;
	}

protected:
	benchmark_problem(float lower_bound, float upper_bound)
	{
		bounds_.lower.fill(lower_bound);
		bounds_.upper.fill(upper_bound);

		for (std::size_t d{}; d < Dim; ++d)
			parameter_names_[d] = "x" + std::to_string(d + 1);
	}

	benchmark_problem(
		const std::array<float, Dim>& lower_bounds,
		const std::array<float, Dim>& upper_bounds)
	{
		bounds_.lower = lower_bounds;
		bounds_.upper = upper_bounds;

		for (std::size_t d{}; d < Dim; ++d)
			parameter_names_[d] = "x" + std::to_string(d + 1);
	}

private:
	optimization_bounds<Dim> bounds_{};
	std::array<std::string, Dim> parameter_names_{};
};

template <std::size_t Dim>
class sphere_problem : public benchmark_problem<Dim>
{
public:
	sphere_problem(float lower = -100.0f, float upper = 100.0f) :
		benchmark_problem<Dim>(lower, upper)
	{}

	float evaluate(const std::array<float, Dim>& x) const override
	{
		constexpr float shift = std::numbers::sqrt2_v<float>;
		float sum{};
		for (std::size_t d{}; d < Dim; ++d)
		{
			const float shifted_x = x[d] - shift;
			sum += shifted_x * shifted_x;
		}

		return sum;
	}
};

template <std::size_t Dim>
class rastrigin_problem : public benchmark_problem<Dim>
{
public:
	rastrigin_problem(float lower = -5.12f, float upper = 5.12f) :
		benchmark_problem<Dim>(lower, upper)
	{}

	float evaluate(const std::array<float, Dim>& x) const override
	{
		constexpr float shift = std::numbers::sqrt2_v<float>;
		constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
		float sum = 10.0f * static_cast<float>(Dim);

		for (std::size_t d{}; d < Dim; ++d)
		{
			const float shifted_x = x[d] - shift;
			sum += shifted_x * shifted_x - 10.0f * std::cos(two_pi * shifted_x);
		}

		return sum;
	}
};

template <std::size_t Dim>
class rosenbrock_problem : public benchmark_problem<Dim>
{
	static_assert(Dim >= 2, "Rosenbrock function requires Dim >= 2");
public:
	rosenbrock_problem(float lower = -5.0f, float upper = 10.0f) :
		benchmark_problem<Dim>(lower, upper)
	{}

	float evaluate(const std::array<float, Dim>& x) const override
	{
		constexpr float shift = std::numbers::sqrt2_v<float> -1.0f;
		float sum{};

		for (std::size_t d{}; d < Dim - 1; ++d)
		{
			const float shifted_xd = x[d] - shift;
			const float shifted_xd1 = x[d + 1] - shift;

			const float t1 = shifted_xd1 - shifted_xd * shifted_xd;
			const float t2 = 1.0f - shifted_xd;
			sum += 100.0f * t1 * t1 + t2 * t2;
		}
		return sum;
	}
};

template <std::size_t Dim>
class ackley_problem : public benchmark_problem<Dim>
{
public:
	ackley_problem(float lower = -32.768f, float upper = 32.768f) :
		benchmark_problem<Dim>(lower, upper)
	{}

	float evaluate(const std::array<float, Dim>& x) const override
	{
		constexpr float shift = std::numbers::sqrt2_v<float>;
		constexpr float a = 20.0f;
		constexpr float b = 0.2f;
		constexpr float c = 2.0f * std::numbers::pi_v<float>;
		constexpr float inv_dim = 1.0f / static_cast<float>(Dim);

		float sum_sq{};
		float sum_cos{};
		for (std::size_t d{}; d < Dim; ++d)
		{
			const float shifted_x = x[d] - shift;
			sum_sq += shifted_x * shifted_x;
			sum_cos += std::cos(c * shifted_x);
		}

		return -a * std::exp(-b * std::sqrt(sum_sq * inv_dim))
			- std::exp(sum_cos * inv_dim)
			+ a + std::numbers::e_v<float>;
	}
};

template <std::size_t Dim>
class bukin6_problem : public benchmark_problem<Dim>
{
public:
	bukin6_problem(
		std::array<float, Dim> lower = { -15.0f, -3.0f },
		std::array<float, Dim> upper = { -5.0f, 3.0f }) :
		benchmark_problem<Dim>(lower, upper)
	{}

	float evaluate(const std::array<float, Dim>& x) const override
	{
		constexpr float shift_x0 = std::numbers::sqrt2_v<float> +10.0f;
		constexpr float shift_x1 = std::numbers::sqrt2_v<float> -1.0f;

		const float shifted_x0 = x[0] - shift_x0;
		const float shifted_x1 = x[1] - shift_x1;

		const float t1 = std::abs(shifted_x1 - 0.01f * shifted_x0 * shifted_x0);
		const float t2 = std::abs(shifted_x0 + 10.0f);
		return 100.0f * std::sqrt(t1) + 0.01f * t2;
	}
};

template <std::size_t Dim>
class beale_problem : public benchmark_problem<Dim>
{
public:
	beale_problem(float lower = -4.5f, float upper = 4.5f) :
		benchmark_problem<Dim>(lower, upper)
	{}

	float evaluate(const std::array<float, Dim>& x) const override
	{
		constexpr float shift_x0 = std::numbers::sqrt2_v<float> -3.0f;
		constexpr float shift_x1 = std::numbers::sqrt2_v<float> -0.5f;

		const float shifted_x1 = x[0] - shift_x0;
		const float shifted_x2 = x[1] - shift_x1;
		const float x2_sq = shifted_x2 * shifted_x2;

		const float t1 = 1.5f - shifted_x1 + shifted_x1 * shifted_x2;
		const float t2 = 2.25f - shifted_x1 + shifted_x1 * x2_sq;
		const float t3 = 2.625f - shifted_x1 + shifted_x1 * x2_sq * shifted_x2;

		return t1 * t1 + t2 * t2 + t3 * t3;
	}
};