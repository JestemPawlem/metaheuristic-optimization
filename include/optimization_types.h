#pragma once

#include <array>
#include <cstddef>


template <std::size_t Dim>
struct optimization_bounds
{
	std::array<float, Dim> lower{};
	std::array<float, Dim> upper{};
};


template <std::size_t Dim>
struct optimization_result
{
	std::size_t n_evaluations{};
	std::array<float, Dim> best_parameters{};
	float best_fitness{};
};


template <std::size_t Dim>
struct iteration_data
{
    std::size_t iteration{};
	std::array<float, Dim> best_parameters{};
    float best_fitness{};
};