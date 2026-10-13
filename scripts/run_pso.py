import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description='Running PSO optimization for IHCP in C++'
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
        '--n-pop',
        str(args.n_pop),
        '--n-iter',
        str(args.n_iter),
        '--inertia',
        str(args.inertia),
        '--c1',
        str(args.c1),
        '--c2',
        str(args.c2)
    ]

    print('Calling PSO...')

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            err_output = process.stderr.read()
            print(f'Error executing C++ program:\n{err_output}', file=sys.stderr)

        end_time = time.perf_counter()
        print(f'Program execution took {end_time - start_time:.2f}s')
    except FileNotFoundError:
        print(f'Executable not found: {exe_path}', file=sys.stderr)


if __name__ == '__main__':
    main()