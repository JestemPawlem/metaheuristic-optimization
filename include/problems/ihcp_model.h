#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include "optimization_problem.h"
#include "optimization_types.h"
#include "random.h"


class ihcp_model : public optimization_problem<2>
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

    std::array<int, num_thermocouples> node_indices = { 451, 401 };
    std::array<float, num_thermocouples> start_times = { 105.0f, 117.0f };

    explicit ihcp_model(
        float noise_level = 0.0f,
        optimization_bounds<2> bounds = {
            { 100'000.0f, 1'000.0f },
            { 600'000.0f, 8'000.0f }
        }) :
        noise_level_(noise_level),
        bounds_(bounds)
    {
        exact_temperatures_.resize(num_thermocouples * num_measurements);
        measured_temperatures_.resize(num_thermocouples * num_measurements);

        std::array<float, 2> true_params{ 400000.0f, 4000.0f };
        generate_reference_data(true_params);
    }

    const optimization_bounds<2>& bounds() const noexcept override
    {
        return bounds_;
    }

    const std::array<std::string, 2>& parameter_names() const noexcept override
    {
        static constexpr std::array<std::string, 2> names{ "q", "alpha" };
        return names;
    }

    float evaluate(const std::array<float, 2>& parameters) const override
    {
        std::vector<std::vector<float>> reconstructed(
            num_thermocouples, std::vector<float>(num_measurements)
        );
        run_simulation(parameters, reconstructed);

        float sq_error = 0.0f;
        for (std::size_t i{}; i < num_thermocouples; ++i)
        {
            for (std::size_t j{}; j < num_measurements; ++j)
            {
                float diff = reconstructed[i][j] - measured_temperatures_[i * num_measurements + j];
                sq_error += diff * diff;
            }
        }

        return (measurement_step * tau) * sq_error;
    }

    std::span<const std::string_view> metric_names() const noexcept override
    {
        static constexpr std::array<std::string_view, 1> names{ "mae" };
        return names;
    }

    std::vector<float> compute_metrics(const std::array<float, 2>& parameters) const override
    {
        std::vector<std::vector<float>> reconstructed(
            num_thermocouples, std::vector<float>(num_measurements)
        );
        run_simulation(parameters, reconstructed);

        float sum_abs_err = 0.0f;
        const std::size_t total_points = num_thermocouples * num_measurements;

        for (std::size_t i{}; i < num_thermocouples; ++i)
        {
            for (std::size_t j{}; j < num_measurements; ++j)
            {
                float diff = std::abs(reconstructed[i][j] - measured_temperatures_[i * num_measurements + j]);
                sum_abs_err += diff;
            }
        }

        return { sum_abs_err / static_cast<float>(total_points) };
    }

    const std::vector<float>& exact_temperatures() const noexcept { return exact_temperatures_; }
    const std::vector<float>& measured_temperatures() const noexcept { return measured_temperatures_; }

private:
    void thomas_algorithm(
        int dim,
        const std::vector<float>& a,
        const std::vector<float>& b,
        const std::vector<float>& c,
        const std::vector<float>& d,
        std::vector<float>& beta,
        std::vector<float>& gamma,
        std::vector<float>& wx) const
    {
        if (b[1] == 0.0f)
            return;

        beta[1] = -c[1] / b[1];
        gamma[1] = d[1] / b[1];

        for (int j = 2; j <= dim - 1; ++j)
        {
            float inv_denom = 1.0f / (a[j] * beta[j - 1] + b[j]);
            beta[j] = -c[j] * inv_denom;
            gamma[j] = (d[j] - a[j] * gamma[j - 1]) * inv_denom;
        }

        float last_denom = a[dim] * beta[dim - 1] + b[dim];
        if (std::abs(last_denom) < 1e-12f)
            last_denom = 1e-12f;

        wx[dim] = (d[dim] - a[dim] * gamma[dim - 1]) / last_denom;

        for (int j = dim - 1; j >= 1; --j)
            wx[j] = beta[j] * wx[j + 1] + gamma[j];
    }

    float temperature_from_enthalpy(float enthalpy, float H1, float H2, float time) const
    {
        if (enthalpy <= H1)
            return enthalpy / (c2 * rho2);

        if (enthalpy >= H2)
        {
            float kappa = (time >= 0.0f) ? L * rho2 : 0.0f;
            return (enthalpy + (c1 * rho1 - c2 * rho2) * solid_temp - kappa) / (c1 * rho1);
        }
        return solid_temp;
    }

    float d_i(int stage, float ent, float H1, float param1, float param2, float time) const
    {
        float p = (stage == 1) ? param1 : param2;
        if (ent <= H1)
            return 0.0f;

        float kappa = (time >= 0.0f) ? L * rho2 : 0.0f;
        float val = ((param1 - param2) * solid_temp + kappa) / param1;
        return p * (ambient_temp - val);
    }

    void run_simulation(
        const std::array<float, 2>& params,
        std::vector<std::vector<float>>& out_reconstructed) const
    {
        float param1 = c1 * rho1;
        float param2 = c2 * rho2;
        float a1 = lambda1 / param1;
        float a2 = lambda2 / param2;
        float dx = (end_pos - start_pos) / static_cast<float>(quantity - 1);

        int size = static_cast<int>(quantity + 2);
        std::vector<float> enthalpy(size);
        std::vector<float> u(size);
        std::vector<float> r(size);
        std::vector<float> temp(size, pouring_temp);
        std::vector<float> aa(size);
        std::vector<float> bb(size);
        std::vector<float> cc(size);
        std::vector<float> ff(size);
        std::vector<float> beta(size);
        std::vector<float> gamma(size);

        float time = 0.0f;
        float H1 = param2 * solid_temp;
        float H2 = H1 + (L * rho2);

        for (std::size_t i = 1; i <= quantity; ++i)
            enthalpy[i] = H2;

        std::array<std::size_t, num_thermocouples> executed{};
        for (std::size_t i = 0; i < num_thermocouples; ++i)
            for (std::size_t j = 0; j < num_measurements; ++j)
                out_reconstructed[i][j] = pouring_temp;

        float max_time = 0.0f;
        for (std::size_t i = 0; i < num_thermocouples; ++i)
        {
            float t_end = start_times[i] + static_cast<float>(num_measurements - 1) * measurement_step * tau;
            if (t_end > max_time)
                max_time = t_end;
        }

        float qq_val = params[0];
        float alfa_val = params[1];
        int flag_phase = 0;

        while (time <= max_time + 1.0f)
        {
            time += tau;

            for (std::size_t i = 1; i <= quantity; ++i)
                u[i] = (enthalpy[i] <= H2) ? H2 : enthalpy[i];

            for (std::size_t i = 1; i <= quantity; ++i)
                r[i] = enthalpy[i] - u[i];

            float ul1 = (dx * dx) / tau;
            aa[1] = 0.0f;
            bb[1] = (dx / (2.0f * a1 * tau)) + (1.0f / dx);
            cc[1] = -(1.0f / dx);
            ff[1] = (dx / (2.0f * a1 * tau)) * u[1];

            for (std::size_t i = 2; i <= quantity - 1; ++i)
            {
                aa[i] = -1.0f;
                cc[i] = -1.0f;
                bb[i] = (ul1 / a1) + 2.0f;
                ff[i] = (ul1 * u[i]) / a1;
            }

            float aq_i = (time > tx1) ? alfa_val : qq_val;
            if (enthalpy[quantity] < H2)
                aq_i = 0.0f;
            else
                flag_phase = 1;

            if (time < tx1)
            {
                aa[quantity] = -(1.0f / dx);
                bb[quantity] = (dx / (2.0f * a1 * tau)) + (1.0f / dx);
                cc[quantity] = 0.0f;
                ff[quantity] = (dx / (2.0f * a1 * tau)) * u[quantity] - (aq_i / a1);
            }
            else
            {
                aa[quantity] = -(lambda1 / dx);
                bb[quantity] = ((lambda1 * dx) / (2.0f * a1 * tau)) + aq_i + (lambda1 / dx);
                cc[quantity] = 0.0f;
                ff[quantity] = ((lambda1 * dx) / (2.0f * a1 * tau)) * u[quantity] + aq_i * d_i(1, enthalpy[quantity], H1, param1, param2, time);
            }

            thomas_algorithm(static_cast<int>(quantity), aa, bb, cc, ff, beta, gamma, u);
            for (std::size_t i = 1; i <= quantity; ++i)
                enthalpy[i] = u[i] + r[i];

            for (std::size_t i = 1; i <= quantity; ++i)
                u[i] = (enthalpy[i] >= H1) ? H1 : enthalpy[i];

            for (std::size_t i = 1; i <= quantity; ++i)
                r[i] = enthalpy[i] - u[i];

            float ul2 = (dx * dx) / tau;
            aa[1] = 0.0f;
            bb[1] = (dx / (2.0f * a2 * tau)) + (1.0f / dx);
            cc[1] = -(1.0f / dx);
            ff[1] = (dx / (2.0f * a2 * tau)) * u[1];

            for (std::size_t i = 2; i <= quantity - 1; ++i)
            {
                aa[i] = -1.0f;
                cc[i] = -1.0f;
                bb[i] = (ul2 / a2) + 2.0f;
                ff[i] = (ul2 * u[i]) / a2;
            }

            aq_i = (time > tx1) ? alfa_val : qq_val;
            if (flag_phase == 1)
            {
                aq_i = 0.0f;
                flag_phase = 0;
            }

            if (time < tx1)
            {
                aa[quantity] = -(1.0f / dx);
                bb[quantity] = (dx / (2.0f * a2 * tau)) + (1.0f / dx);
                cc[quantity] = 0.0f;
                ff[quantity] = (dx / (2.0f * a2 * tau)) * u[quantity] - (aq_i / a2);
            }
            else
            {
                aa[quantity] = -(lambda2 / dx);
                bb[quantity] = ((lambda2 * dx) / (2.0f * a2 * tau)) + aq_i + (lambda2 / dx);
                cc[quantity] = 0.0f;
                ff[quantity] = ((lambda2 * dx) / (2.0f * a2 * tau)) * u[quantity] + aq_i * d_i(2, enthalpy[quantity], H1, param1, param2, time);
            }

            thomas_algorithm(static_cast<int>(quantity), aa, bb, cc, ff, beta, gamma, u);
            for (std::size_t i = 1; i <= quantity; ++i)
                enthalpy[i] = u[i] + r[i];

            for (std::size_t i = 1; i <= quantity; ++i)
                temp[i] = temperature_from_enthalpy(enthalpy[i], H1, H2, time);

            for (std::size_t t_idx = 0; t_idx < num_thermocouples; ++t_idx)
            {
                for (std::size_t p_idx = executed[t_idx]; p_idx < num_measurements; ++p_idx)
                {
                    float read_time = start_times[t_idx] + static_cast<float>(p_idx * measurement_step) * tau;
                    if (std::abs(time - read_time) < (tau / 2.0f))
                    {
                        int node_idx = node_indices[t_idx];
                        if (node_idx >= 1 && node_idx <= static_cast<int>(quantity))
                            out_reconstructed[t_idx][p_idx] = temp[node_idx];
                        executed[t_idx] = p_idx + 1;
                    }
                }
            }
        }
    }

    void generate_reference_data(const std::array<float, 2>& true_params)
    {
        std::vector<std::vector<float>> sim_output(
            num_thermocouples, std::vector<float>(num_measurements)
        );
        run_simulation(true_params, sim_output);

        for (std::size_t i = 0; i < num_thermocouples; ++i)
        {
            for (std::size_t j = 0; j < num_measurements; ++j)
            {
                std::size_t idx = i * num_measurements + j;
                float t_exact = sim_output[i][j];

                exact_temperatures_[idx] = t_exact;
                measured_temperatures_[idx] = (noise_level_ > 0.0f) ?
                    t_exact * rng::normal(1.0f, noise_level_ / 3.0f) :
                    t_exact;
            }
        }
    }

    float noise_level_;
    optimization_bounds<2> bounds_;
    std::vector<float> exact_temperatures_;
    std::vector<float> measured_temperatures_;
};