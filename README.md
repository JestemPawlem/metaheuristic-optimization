# IHCPOptimization

A C++23 framework and Python scripts for evaluating metaheuristic algorithms on standard benchmark functions and the Inverse Heat Conduction Problem (IHCP).

## Project Overview

The project consists of two main components:
1. **Computational Engine (C++23)** – an implementation of metaheuristics and mathematical models, driven by `.cfg` configuration files and logging convergence histories to CSV files.
2. **Orchestration and Data Analysis (Python)** – CLI scripts for single runs, multi-process batch experiment execution, and generation of convergence plots and statistical tables.

### Inverse Heat Conduction Problem (IHCP)
The `ihcp` model simulates 1D casting solidification. The optimization goal is to reconstruct two boundary parameters from temperature readings recorded by 2 thermocouples (100 measurements each, with optional Gaussian measurement noise):
* **`q`** – heat flux density extracted during the first cooling stage ($t \le 110\text{ s}$), reference value: $400\,000$, search bounds: $[100\,000,\; 600\,000]$.
* **`alpha`** – convective heat transfer coefficient during the second cooling stage ($t > 110\text{ s}$), reference value: $4\,000$, search bounds: $[1\,000,\; 8\,000]$.

The objective function is the weighted sum of squared temperature reconstruction errors at the measurement points (an additional Mean Absolute Error metric is also computed).

### Benchmark Functions
Standard test functions are supported across dimensions 2D to 10D (e.g., `sphere2`, `sphere5`, `sphere10`):
* `sphere`, `rosenbrock`, `rastrigin`, `ackley` (2D–10D)
* `bukin` (Bukin N.6, 2D), `beale` (Beale, 2D)

### Implemented Algorithms
* **ACO** (`aco`) – Ant Colony Optimization (continuous domain)
* **AOA** (`aoa`) – Archimedes Optimization Algorithm
* **CS** (`cs`) – Cuckoo Search
* **PSO** (`pso`) – Particle Swarm Optimization
* **RCGA** (`rcga`) – Real-Coded Genetic Algorithm
* **SCA** (`sca`) – Sine Cosine Algorithm

## Requirements

* A compiler supporting the **C++23** standard
* **Python 3.12+** and the **[uv](https://docs.astral.sh/uv/)** package manager

## Usage Instructions

Before running the scripts, ensure the C++ project has been built in `Release` mode.

All commands below should be executed from the `scripts/` directory. Dependencies are automatically resolved and installed by `uv` on the first run.

### 1. Single Run (`run_single.py`)

Generates a `.cfg` configuration file, invokes the C++ executable, and stores the outputs in a session directory:

```bash
# Solve the IHCP problem using PSO (10 runs, population size 30, 100 iterations)
uv run ./run_single.py --problem ihcp --optimizer pso --n-runs 10 --n-solutions 30 --n-iter 100 --noise-level 0.03
```

### 2. Parallel Batch Experiments (`run_experiments.py`)

Executes a parameter set (varying population and iteration sizes) in parallel across the specified problems and optimizers:

```bash
# Run benchmarks
uv run ./run_experiments.py \
    --problems sphere2 rosenbrock5 rastrigin10 bukin \
    --optimizers pso aoa \
    --n-runs 20 \
    --quiet

# Run IHCP
uv run ./run_experiments.py \
    --problems ihcp \
    --optimizers pso \
    --noise-level 0.03
```

Results for each task are saved inside an automatically generated `results/session_YYYYMMDD_HHMMSS/` directory.

### 3. Result Aggregation and Plotting (`plot_benchmarks.py`)

Processes `run_*.csv` files from the latest (or a specified) experimental session:

```bash
# Analyze the most recent session in results/
uv run ./plot_benchmarks.py

# Analyze a specific session and save outputs to a custom directory
uv run ./plot_benchmarks.py --session-dir ../results/session_20261116_203510 --output-dir ./report_plots
```

Generated artifacts (by default in `results/<session>/artifacts/`):
* `aggregated_results.csv` – merged raw data across all runs in the session.
* `clean_statistics.csv` – summary table of final statistics ($f_{\min}$, $f_{\max}$, $\bar{f}$, $f_{\text{med}}$, $s_f$).
* `final_statistics_table.png` – formatted statistical summary rendered as an image.
* `<problem>_fitness_vs_evaluations.png` – convergence curves (median and 25th–75th interquartile range on a logarithmic scale).