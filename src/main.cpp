#include <cstddef>
#include <cstdlib>
#include <print>
#include <string>
#include <vector>

#include "color.h"
#include "ihcp_model.h"
#include "pso.h"


int main(int argc, char* argv[])
{
	std::system("chcp 65001 > nul");

	int pop_size = 30;
	int iterations = 30;
	float inertia = 0.7f;
	float c1 = 1.5f;
	float c2 = 1.5f;
	std::vector<float> lower_bounds = { 100'000.0f, 1'000.0f };
	std::vector<float> upper_bounds = { 600'000.0f, 8'000.0f };

	for (int i = 1; i < argc - 1; i += 2)
	{
		std::string arg = argv[i];
		if (arg == "--n-pop") pop_size = std::stoi(argv[i + 1]);
		else if (arg == "--n-iter") iterations = std::stoi(argv[i + 1]);
		else if (arg == "--inertia") inertia = std::stof(argv[i + 1]);
		else if (arg == "--c1") c1 = std::stof(argv[i + 1]);
		else if (arg == "--c2") c2 = std::stof(argv[i + 1]);
	}

	IHCPModel model;
	ParticleSwarmOptimizer pso(
		pop_size,
		iterations,
		inertia,
		c1,
		c2,
		lower_bounds,
		upper_bounds
	);

	std::println("{}Running PSO optimization for IHCP...{}", color::cyan, color::reset);
	PSOResult result = pso.optimize(model, 2);

	std::println("{}PSO optimization finished.{}", color::cyan, color::reset);
	std::println("Optimal heat flux (q):                 {}{:.2f}{}", color::yellow, result.best_position[0], color::reset);
	std::println("Optimal heat transfer coeff. (alpha):  {}{:.2f}{}", color::yellow, result.best_position[1], color::reset);
	std::println("Objective function value (Fitness):    {:.4f}", result.best_fitness);
	std::println("Achieved mean error:                   {:.6f} K", result.best_mean_error);

	return 0;
}