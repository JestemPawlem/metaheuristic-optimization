import argparse
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.append(str(Path(__file__).resolve().parent))
from common import print_error, print_info, ROOT_DIR, EXE_PATH


def generate_run_id() -> str:
    return uuid.uuid4().hex[:6]

def generate_session_dir() -> Path:
    return ROOT_DIR / 'results' / f'results_{uuid.uuid4().hex[:6]}'


def main():
    parser = argparse.ArgumentParser(
        description='Runner for single IHCP optimization experiment in C++'
    )

    parser.add_argument('--session-dir', type=str)
    parser.add_argument('--run-id', type=str)
    parser.add_argument('--optimizer', type=str, default='pso', choices=['pso', 'sga'])
    parser.add_argument('--n-runs', type=int, default=1)

    parser.add_argument('--n-pop', type=int, default=30)
    parser.add_argument('--n-iter', type=int, default=30)

    parser.add_argument('--inertia', type=float, default=0.7)
    parser.add_argument('--c1', type=float, default=1.5)
    parser.add_argument('--c2', type=float, default=1.5)

    parser.add_argument('--crossover-rate', type=float, default=0.8)
    parser.add_argument('--mutation-rate', type=float, default=0.1)

    args = parser.parse_args()

    session_dir = (Path(args.session_dir).resolve()
                   if args.session_dir
                   else generate_session_dir())
    session_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id if args.run_id else generate_run_id()
    log_file_path = session_dir / f'run_{run_id}.log'

    cmd = [
        str(EXE_PATH),
        str(session_dir),
        str(run_id),
        str(args.optimizer),
        str(args.n_runs),
        str(args.n_pop),
        str(args.n_iter)
    ]

    if args.optimizer == 'pso':
        cmd.extend([str(args.inertia), str(args.c1), str(args.c2)])
    elif args.optimizer == 'sga':
        cmd.extend([str(args.crossover_rate), str(args.mutation_rate)])

    try:
        start_time = time.perf_counter()
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        end_time = time.perf_counter()
        print_info(f'Experiment took {end_time - start_time:.2f}s')

        if result.returncode != 0:
            print_error(
                f'[{args.optimizer.upper()}] FAILED run_{run_id} (code {result.returncode})! Details: {log_file_path.name}',
                file=sys.stderr
            )
            sys.exit(result.returncode)

    except FileNotFoundError:
        print_error(f'Executable not found at path: {EXE_PATH}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()