import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import OPTIMIZERS, ROOT_DIR, print_error, print_info

LOG_LINE_RE = re.compile(r'Iter:\s*(\d+)\s*\|\s*Fitness:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')


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


def parse_log_file(
    log_path: Path,
    problem: str,
    algorithm: str,
    n_solutions: int,
) -> pd.DataFrame:
    rows = []
    run_idx = 0
    prev_iter = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            match = LOG_LINE_RE.search(line)
            if not match:
                continue

            iteration = int(match.group(1))
            fitness = float(match.group(2))

            if prev_iter is not None and iteration <= prev_iter:
                run_idx += 1
            prev_iter = iteration

            rows.append(
                {
                    'problem': problem,
                    'algorithm': algorithm,
                    'run': run_idx,
                    'iteration': iteration,
                    'n_evaluations': (iteration + 1) * n_solutions,
                    'fitness': fitness,
                }
            )

    if not rows:
        raise ValueError(f'No iteration entries parsed from: {log_path.name}')

    return pd.DataFrame(rows)


def aggregate_convergence_data(
    session_dir: Path,
    run_meta: dict[str, dict[str, str]],
    n_solutions: int,
) -> pd.DataFrame:
    log_files = sorted(session_dir.glob('run_*.log'))
    if not log_files:
        raise FileNotFoundError(f'No run_*.log files found in: {session_dir}')

    frames = []
    for log_path in log_files:
        run_id = log_path.stem.replace('run_', '', 1)
        meta = run_meta.get(run_id, {'problem': 'unknown', 'algorithm': run_id})
        frames.append(
            parse_log_file(log_path, meta['problem'], meta['algorithm'], n_solutions)
        )

    df = pd.concat(frames, ignore_index=True)
    aggregated_path = session_dir / 'convergence_aggregated.csv'
    df.to_csv(aggregated_path, index=False)
    print_info(f'Parsed {len(df)} rows from {len(log_files)} files -> {aggregated_path.name}')

    return df


def setup_plot_style() -> None:
    sns.set_theme(style='whitegrid')
    plt.rcParams.update(
        {
            'font.sans-serif': 'DejaVu Sans',
            'axes.edgecolor': '#cccccc',
            'axes.linewidth': 0.8,
        }
    )


def plot_convergence(df: pd.DataFrame, output_dir: Path) -> None:
    df = df.copy()
    df['fitness'] = df['fitness'].clip(lower=1e-6)

    setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    for problem_name, problem_df in df.groupby('problem'):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.lineplot(
            data=problem_df,
            x='n_evaluations',
            y='fitness',
            hue='algorithm',
            ax=ax,
            estimator='median',
            errorbar=('pi', 50),
            linewidth=2.0,
        )
        ax.set_yscale('log')
        ax.set_title(
            f'Convergence Curve ({problem_name})',
            fontsize=12,
            fontweight='bold',
        )
        ax.set_xlabel('Evaluations (NFE)', fontsize=10)
        ax.set_ylabel('Best-so-far Fitness', fontsize=10)
        fig.tight_layout()

        plot_path = output_dir / f'{problem_name}_convergence.png'
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print_info(f'Generated convergence plot: {plot_path.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Parallel convergence analysis runner')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--n-runs', type=int, default=30)
    parser.add_argument('--problems', nargs='+', default=['ihcp'])
    parser.add_argument('--optimizers', nargs='+', default=['aoa'], choices=OPTIMIZERS)
    parser.add_argument('--noise-level', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-solutions', type=int, default=20)
    parser.add_argument('--n-iter', type=int, default=200)

    args, unknown_args = parser.parse_known_args()

    runner_script = ROOT_DIR / 'scripts' / 'run_single.py'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (ROOT_DIR / 'results' / f'conv_session_{timestamp}').resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    run_meta: dict[str, dict[str, str]] = {}
    task_idx = 1

    for problem in args.problems:
        for optimizer in args.optimizers:
            run_id = f'{task_idx:03d}'
            run_meta[run_id] = {'problem': problem, 'algorithm': optimizer}

            task_cmd = [
                str(runner_script),
                '--session-dir', str(session_dir),
                '--run-id', run_id,
                '--seed', str(args.seed + task_idx),
                '--problem', problem,
                '--noise-level', str(args.noise_level),
                '--optimizer', optimizer,
                '--n-runs', str(args.n_runs),
                '--n-solutions', str(args.n_solutions),
                '--n-iter', str(args.n_iter),
                *unknown_args,
            ]
            tasks.append(task_cmd)
            task_idx += 1

    print_info(f'Convergence Session: {session_dir.name}')
    print_info(f"Problems: {', '.join(args.problems)}")
    print_info(f"Optimizers: {', '.join(args.optimizers)}")
    print_info(f'Queued {len(tasks)} tasks...\n')

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]

        with tqdm(
            total=len(tasks),
            desc='Convergence',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
        ) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update()

    elapsed = time.perf_counter() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print_info(f'\nConvergence runs took {minutes:d}m {seconds:d}s')

    try:
        df = aggregate_convergence_data(session_dir, run_meta, args.n_solutions)
        artifacts_dir = session_dir / 'artifacts'
        plot_convergence(df, artifacts_dir)
        print_info(f'Artifacts saved in: {artifacts_dir.resolve()}')
    except Exception as e:
        print_error(f'Failed to generate plots: {e}')


if __name__ == '__main__':
    main()