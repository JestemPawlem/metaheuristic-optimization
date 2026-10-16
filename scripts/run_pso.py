import argparse
import os
import subprocess
import sys
import time


COLOR_MSG = '\033[1;95m'
COLOR_RESET = '\033[0m'


def print_msg(message: str, **kwargs):
    print(f'{COLOR_MSG}{message}{COLOR_RESET}', **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Running PSO optimization for IHCP in C++'
    )

    parser.add_argument(
            '--n-runs',
            type=int,
            default=1,
            help='Number of runs (default: 1)'
        )
    parser.add_argument(
        '--n-pop',
        type=int,
        default=30,
        help='Population size (default: 30)'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=30,
        help='Number of iterations (default: 30)'
    )
    parser.add_argument(
        '--inertia',
        type=float,
        default=0.7,
        help='Inertia w (default: 0.7)'
    )
    parser.add_argument(
        '--c1',
        type=float,
        default=1.5,
        help='Coefficient c1 (default: 1.5)'
    )
    parser.add_argument(
        '--c2',
        type=float,
        default=1.5,
        help='Coefficient c2 (default: 1.5)'
    )

    args = parser.parse_args()

    exe_path = os.path.join('..', 'out', 'build', 'x64-Release', 'ihcp_pso.exe')

    cmd = [
        exe_path,
        str(args.n_runs),
        str(args.n_pop),
        str(args.n_iter),
        str(args.inertia),
        str(args.c1),
        str(args.c2)
    ]

    print_msg('Calling PSO...')

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            err_output = process.stderr.read()
            print_msg(f'Error executing C++ program:\n{err_output}', file=sys.stderr)

        end_time = time.perf_counter()
        print_msg(f'Program execution took {end_time - start_time:.2f}s')
    except FileNotFoundError:
        print_msg(f'Executable not found: {exe_path}', file=sys.stderr)


if __name__ == '__main__':
    main()