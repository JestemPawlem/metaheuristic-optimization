#include "ihcp_model.h"

IHCPModel::IHCPModel() {
    exact_temperatures.resize(num_thermocouples * num_measurements, 0.0f);
    std::vector<float> true_params = { 400000.0f, 4000.0f };
    generate_reference_data(true_params);
}

void IHCPModel::thomas_algorithm(int dim, const std::vector<float>& a, const std::vector<float>& b,
    const std::vector<float>& c, const std::vector<float>& d, std::vector<float>& wx) const {
    int size = static_cast<int>(wx.size());
    std::vector<float> beta(size, 0.0f);
    std::vector<float> gamma(size, 0.0f);

    if (b[1] == 0.0f) return;
    beta[1] = -c[1] / b[1];
    gamma[1] = d[1] / b[1];

    for (int j = 2; j <= dim - 1; ++j) {
        float denom = a[j] * beta[j - 1] + b[j];
        if (std::abs(denom) < 1e-12f) denom = 1e-12f;
        beta[j] = -c[j] / denom;
        gamma[j] = (d[j] - a[j] * gamma[j - 1]) / denom;
    }

    float last_denom = (a[dim] * beta[dim - 1] + b[dim]);
    if (std::abs(last_denom) < 1e-12f) last_denom = 1e-12f;
    wx[dim] = (d[dim] - a[dim] * gamma[dim - 1]) / last_denom;

    for (int j = dim - 1; j >= 1; --j) {
        wx[j] = beta[j] * wx[j + 1] + gamma[j];
    }
}

float IHCPModel::temperature_from_enthalpy(float enthalpy, float H1, float H2, float time) const {
    if (enthalpy <= H1) return enthalpy / (c2 * rho2);
    if (enthalpy >= H2) {
        float kappa = (time >= 0.0f) ? L * rho2 : 0.0f;
        return (enthalpy + (c1 * rho1 - c2 * rho2) * solid_temp - kappa) / (c1 * rho1);
    }
    return solid_temp;
}

float IHCPModel::d_i(int stage, float ent, float H1, float param1, float param2, float time) const {
    float p = (stage == 1) ? param1 : param2;
    if (ent <= H1) return 0.0f;
    float kappa = (time >= 0.0f) ? L * rho2 : 0.0f;
    float val = ((param1 - param2) * solid_temp + kappa) / param1;
    return p * (ambient_temp - val);
}

void IHCPModel::run_simulation(const std::vector<float>& parameters_q, std::vector<std::vector<float>>& out_reconstructed) const {
    float param1 = c1 * rho1;
    float param2 = c2 * rho2;
    float a1 = lambda1 / param1;
    float a2 = lambda2 / param2;
    float dx = (end_pos - start_pos) / (quantity - 1);

    int sz = quantity + 2;
    std::vector<float> enthalpy(sz, 0.0f);
    std::vector<float> u(sz, 0.0f);
    std::vector<float> r(sz, 0.0f);
    std::vector<float> temp(sz, pouring_temp);
    std::vector<float> aa(sz, 0.0f);
    std::vector<float> bb(sz, 0.0f);
    std::vector<float> cc(sz, 0.0f);
    std::vector<float> ff(sz, 0.0f);

    float time = 0.0f;
    float H1 = param2 * solid_temp;
    float H2 = H1 + (L * rho2);

    for (int i = 1; i <= quantity; ++i) enthalpy[i] = H2;

    int executed[num_thermocouples] = { 0 };
    for (int i = 0; i < num_thermocouples; ++i) {
        for (int j = 0; j < num_measurements; ++j) {
            out_reconstructed[i][j] = pouring_temp;
        }
    }

    float max_time = 0.0f;
    for (int i = 0; i < num_thermocouples; ++i) {
        float t_end = start_times[i] + (num_measurements - 1) * measurement_step * tau;
        if (t_end > max_time) max_time = t_end;
    }

    float qq_val = parameters_q[0];
    float alfa_val = parameters_q[1];
    int flag_phase = 0;

    while (time <= max_time + 1.0f) {
        time += tau;

        for (int i = 1; i <= quantity; ++i) u[i] = (enthalpy[i] <= H2) ? H2 : enthalpy[i];
        for (int i = 1; i <= quantity; ++i) r[i] = enthalpy[i] - u[i];

        float ul1 = (dx * dx) / tau;
        aa[1] = 0.0f; bb[1] = (dx / (2.0f * a1 * tau)) + (1.0f / dx); cc[1] = -(1.0f / dx);
        ff[1] = (dx / (2.0f * a1 * tau)) * u[1];

        for (int i = 2; i <= quantity - 1; ++i) {
            aa[i] = -1.0f; cc[i] = -1.0f; bb[i] = (ul1 / a1) + 2.0f;
            ff[i] = (ul1 * u[i]) / a1;
        }

        float aq_i = (time > tx1) ? alfa_val : qq_val;
        if (enthalpy[quantity] < H2) aq_i = 0.0f; else flag_phase = 1;

        if (time < tx1) {
            aa[quantity] = -(1.0f / dx); bb[quantity] = (dx / (2.0f * a1 * tau)) + (1.0f / dx); cc[quantity] = 0.0f;
            ff[quantity] = (dx / (2.0f * a1 * tau)) * u[quantity] - aq_i / a1;
        }
        else {
            aa[quantity] = -(lambda1 / dx);
            bb[quantity] = ((lambda1 * dx) / (2.0f * a1 * tau)) + aq_i + (lambda1 / dx);
            cc[quantity] = 0.0f;
            ff[quantity] = ((lambda1 * dx) / (2.0f * a1 * tau)) * u[quantity] + aq_i * d_i(1, enthalpy[quantity], H1, param1, param2, time);
        }

        thomas_algorithm(quantity, aa, bb, cc, ff, u);
        for (int i = 1; i <= quantity; ++i) enthalpy[i] = u[i] + r[i];

        for (int i = 1; i <= quantity; ++i) u[i] = (enthalpy[i] >= H1) ? H1 : enthalpy[i];
        for (int i = 1; i <= quantity; ++i) r[i] = enthalpy[i] - u[i];

        float ul2 = (dx * dx) / tau;
        aa[1] = 0.0f; bb[1] = (dx / (2.0f * a2 * tau)) + (1.0f / dx); cc[1] = -(1.0f / dx);
        ff[1] = (dx / (2.0f * a2 * tau)) * u[1];

        for (int i = 2; i <= quantity - 1; ++i) {
            aa[i] = -1.0f; cc[i] = -1.0f; bb[i] = (ul2 / a2) + 2.0f;
            ff[i] = (ul2 * u[i]) / a2;
        }

        aq_i = (time > tx1) ? alfa_val : qq_val;
        if (flag_phase == 1) { aq_i = 0.0f; flag_phase = 0; }

        if (time < tx1) {
            aa[quantity] = -(1.0f / dx); bb[quantity] = (dx / (2.0f * a2 * tau)) + (1.0f / dx); cc[quantity] = 0.0f;
            ff[quantity] = (dx / (2.0f * a2 * tau)) * u[quantity] - aq_i / a2;
        }
        else {
            aa[quantity] = -(lambda2 / dx);
            bb[quantity] = ((lambda2 * dx) / (2.0f * a2 * tau)) + aq_i + (lambda2 / dx);
            cc[quantity] = 0.0f;
            ff[quantity] = ((lambda2 * dx) / (2.0f * a2 * tau)) * u[quantity] + aq_i * d_i(2, enthalpy[quantity], H1, param1, param2, time);
        }

        thomas_algorithm(quantity, aa, bb, cc, ff, u);
        for (int i = 1; i <= quantity; ++i) enthalpy[i] = u[i] + r[i];

        for (int i = 1; i <= quantity; ++i)
            temp[i] = temperature_from_enthalpy(enthalpy[i], H1, H2, time);

        for (int t_idx = 0; t_idx < num_thermocouples; ++t_idx) {
            for (int p_idx = executed[t_idx]; p_idx < num_measurements; ++p_idx) {
                float read_time = start_times[t_idx] + p_idx * measurement_step * tau;
                if (std::abs(time - read_time) < (tau / 2.0f)) {
                    int node_idx = node_indices[t_idx];
                    if (node_idx >= 1 && node_idx <= quantity) {
                        out_reconstructed[t_idx][p_idx] = temp[node_idx];
                    }
                    executed[t_idx] = p_idx + 1;
                }
            }
        }
    }
}

void IHCPModel::generate_reference_data(const std::vector<float>& true_params) {
    std::vector<std::vector<float>> temp_reconstructed(num_thermocouples, std::vector<float>(num_measurements, 0.0f));
    run_simulation(true_params, temp_reconstructed);
    for (int i = 0; i < num_thermocouples; ++i) {
        for (int j = 0; j < num_measurements; ++j) {
            exact_temperatures[i * num_measurements + j] = temp_reconstructed[i][j];
        }
    }
}

OptimizationResult IHCPModel::objective_function(const std::vector<float>& parameters_q) const {
    std::vector<std::vector<float>> reconstructed(num_thermocouples, std::vector<float>(num_measurements, 0.0f));
    run_simulation(parameters_q, reconstructed);

    float sq_error = 0.0f;
    float max_err = 0.0f;
    float sum_abs_err = 0.0f;
    int total_points = num_thermocouples * num_measurements;

    for (int i = 0; i < num_thermocouples; ++i) {
        for (int j = 0; j < num_measurements; ++j) {
            float diff = reconstructed[i][j] - exact_temperatures[i * num_measurements + j];
            sq_error += diff * diff;

            float abs_diff = std::abs(diff);
            sum_abs_err += abs_diff;
            if (abs_diff > max_err) max_err = abs_diff;
        }
    }

    OptimizationResult res;
    res.fitness = (measurement_step * tau) * sq_error;
    res.max_error = max_err;
    res.mean_error = sum_abs_err / static_cast<float>(total_points);

    return res;
}