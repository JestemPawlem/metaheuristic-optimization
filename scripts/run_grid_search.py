import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import itertools

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import ROOT_DIR, print_info, print_error

VMIN = 1e-6
VMAX = 1


def run_experiment_task(task_args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable] + task_args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
    )
    if result.returncode != 0:
        print_error(f'\nTask execution failed:\n{result.stderr}', file=sys.stderr)
        raise RuntimeError(f'Process returned exit code {result.returncode}')


def read_cfg_params(cfg_path: Path, target_params: list[str]) -> dict[str, float]:
    params = {}
    if not cfg_path.exists():
        return params

    normalized_targets = {p.replace('.', '-').replace('_', '-'): p for p in target_params}

    for line in cfg_path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            key, val = (part.strip() for part in line.split('=', 1))
            norm_key = key.replace('.', '-').replace('_', '-')
            if norm_key in normalized_targets:
                params[normalized_targets[norm_key]] = float(val)
    return params


def aggregate_grid_data(session_dir: Path, target_params: list[str]) -> pd.DataFrame:
    csv_files = sorted(session_dir.glob('run_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No run_*.csv files found in: {session_dir}')

    frames = []
    for f in csv_files:
        part = pd.read_csv(f)
        cfg_params = read_cfg_params(f.with_suffix('.cfg'), target_params)
        
        final_row = part.iloc[-1:].copy()
        
        for key, val in cfg_params.items():
            final_row[key] = val
            
        frames.append(final_row)

    df = pd.concat(frames, ignore_index=True)
    
    grouped = df.groupby(target_params)['fitness'].median().reset_index()

    aggregated_path = session_dir / 'grid_aggregated.csv'
    grouped.to_csv(aggregated_path, index=False)
    print_info(f'Aggregated {len(df)} runs into {len(grouped)} grid points -> {aggregated_path.name}')
    
    return grouped


def setup_plot_style():
    sns.set_theme(style='whitegrid')
    plt.rcParams.update(
        {
            'font.sans-serif': 'DejaVu Sans',
            'axes.edgecolor': '#cccccc',
            'axes.linewidth': 0.8,
        }
    )


def plot_grid_heatmaps(df: pd.DataFrame, output_dir: Path, problem_name: str, x_param: str, y_param: str, z_param: str | None) -> None:
    df['fitness'] = df['fitness'].clip(lower=VMIN, upper=VMAX)
    setup_plot_style()

    if z_param:
        z_values = sorted(df[z_param].unique())
        n_cols = len(z_values)
    else:
        z_values = [None]
        n_cols = 1

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), constrained_layout=True)
    
    if n_cols == 1:
        axes = [axes]
        
    norm = LogNorm(vmin=VMIN, vmax=VMAX)

    for i, z_val in enumerate(z_values):
        ax = axes[i]
        
        if z_param:
            df_slice = df[df[z_param] == z_val]
            title = f'{z_param} = {z_val:g}'
        else:
            df_slice = df
            title = f'Grid Search: {x_param} vs {y_param}'
        
        heatmap_data = df_slice.pivot(index=y_param, columns=x_param, values='fitness')[::-1]
        heatmap_data.index = heatmap_data.index.astype(float).round(4)
        heatmap_data.columns = heatmap_data.columns.astype(float).round(4)

        sns.heatmap(
            heatmap_data, annot=False, cmap='plasma', ax=ax, norm=norm, cbar=False
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f'{x_param}')

        if i == 0:
            ax.set_ylabel(f'Problem: {problem_name}\n\n{y_param}', fontsize=12)
        else:
            ax.set_ylabel('')

    sm = ScalarMappable(norm=norm, cmap='plasma')
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label='Median Best Fitness')

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f'grid_search_{problem_name}_heatmaps.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print_info(f'Generated grid search heatmap: {plot_path.name}')


def generate_grid_values(center: float, n_points: int) -> list[float]:
    if n_points <= 1:
        return [center]
    step = (2.0 - 0.5) / (n_points - 1)
    multipliers = [0.5 + i * step for i in range(n_points)]
    return [center * m for m in multipliers]


def main():
    parser = argparse.ArgumentParser(description='Generic parallel grid search for optimization hyperparameters')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--problem', type=str, default='ihcp')
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--n-solutions', type=int, default=30)
    parser.add_argument('--n-iter', type=int, default=100)
    
    parser.add_argument('--x-param', type=str, required=True)
    parser.add_argument('--x-val', type=float, required=True)
    parser.add_argument('--x-points', type=int, default=5)
    
    parser.add_argument('--y-param', type=str, required=True)
    parser.add_argument('--y-val', type=float, required=True)
    parser.add_argument('--y-points', type=int, default=5)
    
    parser.add_argument('--z-param', type=str, default=None)
    parser.add_argument('--z-val', type=float, default=None)
    parser.add_argument('--z-points', type=int, default=5)
    
    args = parser.parse_args()

    runner_script = ROOT_DIR / 'scripts' / 'run_single.py'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (ROOT_DIR / 'results' / f'grid_session_{args.optimizer}_{timestamp}').resolve()

    active_params = [args.x_param, args.y_param]
    grid_values = [
        generate_grid_values(args.x_val, args.x_points),
        generate_grid_values(args.y_val, args.y_points)
    ]

    if args.z_param and args.z_val is not None:
        active_params.append(args.z_param)
        grid_values.append(generate_grid_values(args.z_val, args.z_points))

    grid = list(itertools.product(*grid_values))

    tasks = []
    task_idx = 1

    for combo in grid:
        task_cmd = [
            str(runner_script),
            '--session-dir', str(session_dir),
            '--run-id', f'grid_{task_idx:04d}',
            '--seed', str(args.seed + task_idx),
            '--problem', args.problem,
            '--optimizer', args.optimizer,
            '--n-runs', str(args.n_runs),
            '--n-solutions', str(args.n_solutions),
            '--n-iter', str(args.n_iter),
        ]
        
        for param_name, param_val in zip(active_params, combo):
            cli_arg = f'--{param_name.replace(".", "-")}'
            task_cmd.extend([cli_arg, str(param_val)])
            
        if args.quiet:
            task_cmd.append('--quiet')

        tasks.append(task_cmd)
        task_idx += 1

    print_info(f'Grid Search Session: {session_dir.name}')
    print_info(f'Problem: {args.problem} (Pop: {args.n_solutions}, Iter: {args.n_iter})')
    print_info(f'Configurations: {len(grid)} (x{args.n_runs} runs = {len(tasks)} tasks)')

    start_time = time.perf_counter()
    session_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]

        with tqdm(
            total=len(tasks),
            desc='Grid Search',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
        ) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update()

    elapsed = time.perf_counter() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print_info(f'\nGrid Search took {minutes:d}m {seconds:d}s')

    try:
        df = aggregate_grid_data(session_dir, active_params)
        artifacts_dir = session_dir / 'artifacts'
        plot_grid_heatmaps(df, artifacts_dir, args.problem, args.x_param, args.y_param, args.z_param)
        print_info(f'Artifacts saved in: {artifacts_dir.resolve()}')
    except Exception as e:
        print_error(f'Failed to generate plots: {e}')


if __name__ == '__main__':
    main()