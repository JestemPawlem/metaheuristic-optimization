import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import ROOT_DIR, print_info, print_error, OPTIMIZERS


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


def main():
    parser = argparse.ArgumentParser(description='Parallel optimization runner with progress tracking')
    parser.add_argument('--workers', type=int, default=None, help='Worker processes count (default: all CPU cores)')
    parser.add_argument('--n-runs', type=int, default=10, help='Repetitions per configuration (default: 10)')
    parser.add_argument('--problems', nargs='+', default=['ihcp'], help='Problems to benchmark')
    parser.add_argument('--optimizers', nargs='+', default=['pso'], choices=OPTIMIZERS, help='Optimizers to use')
    parser.add_argument('--noise-level', type=float, default=0.03, help='IHCP sensor noise level (default: 0.03)')
    parser.add_argument('--seed', type=int, default=0, help='Base RNG seed (default: 0)')
    parser.add_argument('--quiet', action='store_true', help='Silence C++ output logging')
    args = parser.parse_args()

    runner_script = ROOT_DIR / 'scripts' / 'run_single.py'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = (ROOT_DIR / 'results' / f'session_{timestamp}').resolve()

    configurations = [
        {'n_solutions': 8, 'n_iter': 80},
        {'n_solutions': 12, 'n_iter': 120},
        {'n_solutions': 16, 'n_iter': 160},
        {'n_solutions': 20, 'n_iter': 200},
        {'n_solutions': 24, 'n_iter': 240},
        {'n_solutions': 28, 'n_iter': 280},
        {'n_solutions': 32, 'n_iter': 320},
        {'n_solutions': 36, 'n_iter': 360},
        {'n_solutions': 40, 'n_iter': 400},
    ]

    tasks = []
    task_idx = 1

    for problem in args.problems:
        for optimizer in args.optimizers:
            for configuration in configurations:
                task_cmd = [
                    str(runner_script),
                    '--session-dir',
                    str(session_dir),
                    '--run-id',
                    f'{task_idx:03d}',
                    '--seed',
                    str(args.seed + task_idx),
                    '--problem',
                    problem,
                    '--noise-level',
                    str(args.noise_level),
                    '--optimizer',
                    optimizer,
                    '--n-runs',
                    str(args.n_runs),
                    '--n-solutions',
                    str(configuration['n_solutions']),
                    '--n-iter',
                    str(configuration['n_iter']),
                ]
                if args.quiet:
                    task_cmd.append('--quiet')

                tasks.append(task_cmd)
                task_idx += 1

    print_info(f'Session directory: {session_dir.name}')
    print_info(f"Problems: {', '.join(args.problems)}")
    print_info(f'Queued {len(tasks)} tasks...\n')

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_experiment_task, t) for t in tasks]

        with tqdm(
            total=len(tasks),
            desc='Benchmarking',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
        ) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update()

    elapsed = time.perf_counter() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print_info(f'Experiments took {minutes:d}m {seconds:d}s')


if __name__ == '__main__':
    main()