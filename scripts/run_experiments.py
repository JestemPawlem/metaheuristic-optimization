import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

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
    parser.add_argument('--n-runs', type=int, default=5, help='Repetitions per configuration (default: 10)')
    parser.add_argument('--noise-level', type=float, default=0.00, help='IHCP sensor noise level (default: 0.03)')
    args = parser.parse_args()

    runner_script = ROOT_DIR / 'scripts' / 'run_single.py'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (ROOT_DIR / 'results' / f'session_{timestamp}').resolve()

    configurations = [
        {'n_solutions': 10, 'n_iter': 40},
        {'n_solutions': 15, 'n_iter': 60},
        {'n_solutions': 20, 'n_iter': 80},
        {'n_solutions': 25, 'n_iter': 100},
    ]
    optimizers = [
        'aco', 'cs', 'pso', 'rcga'
    ]

    tasks = []
    task_idx = 1

    for configuration in configurations:
        for optimizer in optimizers:
            task_cmd = [
                str(runner_script),
                '--session-dir', str(session_dir),
                '--run-id', f'{task_idx:02d}',
                '--noise-level', str(args.noise_level),
                '--optimizer', optimizer,
                '--n-runs', str(args.n_runs),
                '--n-solutions', str(configuration['n_solutions']),
                '--n-iter', str(configuration['n_iter'])
            ]

            tasks.append(task_cmd)
            task_idx += 1

    print_info(f'Session directory: {session_dir.name}')
    print_info(f'Queued {len(tasks)} tasks...\n')

    start_time = time.perf_counter()
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


if __name__ == '__main__':
    main()