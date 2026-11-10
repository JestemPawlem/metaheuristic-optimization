#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "ihcp_parameters.h"
#include "ihcp_result.h"


class ihcp_model
{
public:
    static constexpr std::size_t quantity = 501;
    static constexpr std::size_t num_thermocouples = 2;
    static constexpr std::size_t num_measurements = 100;
    static constexpr std::size_t measurement_step = 10;

    float lambda1 = 104.0f;
    float lambda2 = 240.0f;
    float c1 = 1290.0f;
    float c2 = 1000.0f;
    float rho1 = 2380.0f;
    float rho2 = 2679.0f;
    float solid_temp = 930.0f;
    float ambient_temp = 298.0f;
    float pouring_temp = 1013.0f;
    float L = 390000.0f;

    float start_pos = 0.0f;
    float end_pos = 0.1f;
    float tau = 0.1f;
    float tx1 = 110.0f;

    int node_indices[num_thermocouples] = { 451, 401 };
    float start_times[num_thermocouples] = { 105.0f, 117.0f };

    ihcp_model(float noise_level = 0.0f);

    ihcp_result objective_function(const ihcp_parameters& params) const;
    const std::vector<float>& get_exact_temperatures() const noexcept;
    const std::vector<float>& get_measured_temperatures() const noexcept;

private:
    void thomas_algorithm(
        int dim,
        const std::vector<float>& a,
        const std::vector<float>& b,
        const std::vector<float>& c,
        const std::vector<float>& d,
        std::vector<float>& beta,
        std::vector<float>& gamma,
        std::vector<float>& wx) const;

    float temperature_from_enthalpy(float enthalpy, float H1, float H2, float time) const;

    float d_i(int stage, float ent, float H1, float param1, float param2, float time) const;

    void run_simulation(
        const ihcp_parameters& parameters_q,
        std::vector<std::vector<float>>& out_reconstructed) const;

    void generate_reference_data(const ihcp_parameters& true_params);

    float noise_level;

    std::vector<float> exact_temperatures;
    std::vector<float> measured_temperatures;
};