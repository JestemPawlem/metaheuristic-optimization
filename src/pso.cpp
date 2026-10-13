#include "pso.h"

ParticleSwarmOptimizer::ParticleSwarmOptimizer(int pop_size, int iter, float inertia, float c1, float c2, float lb, float ub)
    : population_size(pop_size), num_iterations(iter), w(inertia), c1(c1), c2(c2), lower_bound(lb), upper_bound(ub) {}

PSOResult ParticleSwarmOptimizer::optimize(const IHCPModel& model, int dimensions) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_pos(lower_bound, upper_bound);

    float range = upper_bound - lower_bound;
    std::uniform_real_distribution<float> dist_vel(-range * 0.05f, range * 0.05f);
    std::uniform_real_distribution<float> dist_rand(0.0f, 1.0f);

    std::vector<Particle> swarm(population_size);
    std::vector<float> global_best_position(dimensions);
    float global_best_fitness = std::numeric_limits<float>::max();
    float global_best_mean_error = std::numeric_limits<float>::max();

    float vmax = range * 0.2f;

    for (int i = 0; i < population_size; ++i) {
        swarm[i].position.resize(dimensions);
        swarm[i].velocity.resize(dimensions);
        swarm[i].best_position.resize(dimensions);

        for (int d = 0; d < dimensions; ++d) {
            swarm[i].position[d] = dist_pos(gen);
            swarm[i].velocity[d] = dist_vel(gen);
            swarm[i].best_position[d] = swarm[i].position[d];
        }

        OptimizationResult res = model.objective_function(swarm[i].position);
        swarm[i].best_fitness = res.fitness;
        swarm[i].best_mean_error = res.mean_error;

        if (res.fitness < global_best_fitness) {
            global_best_fitness = res.fitness;
            global_best_mean_error = res.mean_error;
            global_best_position = swarm[i].position;
        }
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int i = 0; i < population_size; ++i) {
            for (int d = 0; d < dimensions; ++d) {
                float r1 = dist_rand(gen);
                float r2 = dist_rand(gen);

                swarm[i].velocity[d] = w * swarm[i].velocity[d] +
                    c1 * r1 * (swarm[i].best_position[d] - swarm[i].position[d]) +
                    c2 * r2 * (global_best_position[d] - swarm[i].position[d]);

                swarm[i].velocity[d] = std::clamp(swarm[i].velocity[d], -vmax, vmax);
                swarm[i].position[d] += swarm[i].velocity[d];

                if (swarm[i].position[d] < lower_bound) {
                    swarm[i].position[d] = lower_bound;
                    swarm[i].velocity[d] = 0.0f;
                }
                else if (swarm[i].position[d] > upper_bound) {
                    swarm[i].position[d] = upper_bound;
                    swarm[i].velocity[d] = 0.0f;
                }
            }

            OptimizationResult res = model.objective_function(swarm[i].position);

            if (res.fitness < swarm[i].best_fitness) {
                swarm[i].best_fitness = res.fitness;
                swarm[i].best_mean_error = res.mean_error;
                swarm[i].best_position = swarm[i].position;

                if (res.fitness < global_best_fitness) {
                    global_best_fitness = res.fitness;
                    global_best_mean_error = res.mean_error;
                    global_best_position = swarm[i].position;
                }
            }
        }

        std::cout << "Iteracja " << iter + 1 << "/" << num_iterations
            << " | Najlepszy Fitness: " << global_best_fitness
            << " | Średni Błąd: " << global_best_mean_error << " K\n";
    }

    PSOResult final_res;
    final_res.best_position = global_best_position;
    final_res.best_fitness = global_best_fitness;
    final_res.best_mean_error = global_best_mean_error;
    return final_res;
}