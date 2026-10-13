#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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
	std::vector<float> lower_bounds = { 0.0f, 0.0f };
	std::vector<float> upper_bounds = { 800'000.0f, 10'000.0f };

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

	std::cout << "Running PSO optimization for IHCP...\n";
	PSOResult result = pso.optimize(model, 2);

	std::cout << "\n=== FINAL PSO OPTIMIZATION RESULTS ===\n";
	std::cout << "Optimal heat flux (q):                 " << result.best_position[0] << std::endl;
	std::cout << "Optimal heat transfer coeff. (alpha):  " << result.best_position[1] << std::endl;
	std::cout << "Objective function value (Fitness):    " << result.best_fitness << std::endl;
	std::cout << "Achieved mean error:                   " << result.best_mean_error << std::endl;

	return 0;
}