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
		float sum{};
		for (std::size_t d{}; d < Dim; ++d)
			sum += x[d] * x[d];

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
		constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
		float sum = 10.0f * static_cast<float>(Dim);

		for (std::size_t d{}; d < Dim; ++d)
			sum += x[d] * x[d] - 10.0f * std::cos(two_pi * x[d]);

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
		float sum{};
		for (std::size_t d{}; d < Dim - 1; ++d)
		{
			const float t1 = x[d + 1] - x[d] * x[d];
			const float t2 = 1.0f - x[d];
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
		constexpr float a = 20.0f;
		constexpr float b = 0.2f;
		constexpr float c = 2.0f * std::numbers::pi_v<float>;
		constexpr float inv_dim = 1.0f / static_cast<float>(Dim);

		float sum_sq{};
		float sum_cos{};
		for (std::size_t d{}; d < Dim; ++d)
		{
			sum_sq += x[d] * x[d];
			sum_cos += std::cos(c * x[d]);
		}

		return -a * std::exp(-b * std::sqrt(sum_sq * inv_dim))
			- std::exp(sum_cos * inv_dim)
			+ a + std::numbers::e_v<float>;
	}
};