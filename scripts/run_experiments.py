import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import ROOT_DIR, print_info, print_error


def run_experiment_task(task_args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable] + task_args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    if result.returncode != 0:
        print_error(f'\nTask execution failed:\n{result.stderr}', file=sys.stderr)
        raise RuntimeError(f'Process returned exit code {result.returncode}')


def main():
    parser = argparse.ArgumentParser(
        description='Parallel IHCP optimization runner with progress tracking'
    )
    parser.add_argument('--workers', type=int, default=None, help='Worker processes count (default: all CPU cores)')
    parser.add_argument('--n-runs', type=int, default=10, help='Repetitions per configuration (default: 10)')
    args = parser.parse_args()

    runner_script = ROOT_DIR / 'scripts' / 'run_single.py'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (ROOT_DIR / 'results' / f'session_{timestamp}').resolve()

    configurations = [
        {'n_pop': 10, 'n_iter': 40},
        {'n_pop': 15, 'n_iter': 60},
        {'n_pop': 20, 'n_iter': 80},
        {'n_pop': 25, 'n_iter': 100},
        {'n_pop': 30, 'n_iter': 120},
    ]
    optimizers = [
        'pso', 'sga'
    ]

    tasks = []
    task_idx = 1

    for configuration in configurations:
        for optimizer in optimizers:
            task_cmd = [
                str(runner_script),
                '--session-dir', str(session_dir),
                '--run-id', f'{task_idx:02d}',
                '--optimizer', optimizer,
                '--n-runs', str(args.n_runs),
                '--n-pop', str(configuration['n_pop']),
                '--n-iter', str(configuration['n_iter'])
            ]

            if optimizer == 'pso':
                task_cmd.extend(['--inertia', '0.7', '--c1', '1.5', '--c2', '1.5'])
            elif optimizer == 'sga':
                task_cmd.extend(['--crossover-rate', '0.8', '--mutation-rate', '0.1'])

            tasks.append(task_cmd)
            task_idx += 1

    print_info(f'Session directory: {session_dir.name}')
    print_info(f'Queued {len(tasks)} tasks...\n')

    start_time = time.perf_counter();
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]

        with tqdm(total=len(tasks),
                  desc='Benchmarking',
                  ncols=80,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update()
    elapsed = time.perf_counter() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print_info(f'Experiments took {minutes:d}m {seconds:d}s')

    csv_files = sorted(session_dir.glob('run_*.csv'))
    if not csv_files:
        print_error('No output CSV files were generated.', file=sys.stderr)
        sys.exit(1)

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    summary_file = session_dir / 'aggregated_results.csv'
    df.to_csv(summary_file, index=False)

    print_info(f'\nAggregated {len(df)} runs from {len(csv_files)} files into {summary_file.name}')


if __name__ == '__main__':
    main()