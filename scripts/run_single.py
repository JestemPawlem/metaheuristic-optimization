import argparse
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.append(str(Path(__file__).resolve().parent))
from common import print_error, print_info, ROOT_DIR, EXE_PATH, OPTIMIZERS


def generate_run_id() -> str:
    return uuid.uuid4().hex[:6]


def generate_session_dir() -> Path:
    return ROOT_DIR / 'results' / f'results_{uuid.uuid4().hex[:6]}'


def write_config(config_path: Path,
                 session_dir: Path,
                 run_id: str,
                 args: argparse.Namespace) -> None:
    lines = [
        f'session_dir = {session_dir}',
        f'run_id = {run_id}',
        f'seed = {args.seed}',
        f'n_runs = {args.n_runs}',
        f'verbose = {'false' if args.quiet else 'true'}',
        '',
        f'problem = {args.problem}',
        f'ihcp.noise = {args.noise_level}',
        '',
        f'algorithm = {args.optimizer}',
        f'n_solutions = {args.n_solutions}',
        f'n_iterations = {args.n_iter}',
        '',
        f'aco.locality = {args.aco_locality}',
        f'aco.evaporation = {args.aco_evaporation}',
        '',
        f'aoa.c1 = {args.aoa_c1}',
        f'aoa.c2 = {args.aoa_c2}',
        f'aoa.c3 = {args.aoa_c3}',
        f'aoa.c4 = {args.aoa_c4}',
        '',
        f'cs.pa = {args.cs_pa}',
        '',
        f'pso.inertia = {args.pso_inertia}',
        f'pso.c1 = {args.pso_c1}',
        f'pso.c2 = {args.pso_c2}',
        '',
        f'rcga.crossover_rate = {args.rcga_crossover_rate}',
        f'rcga.mutation_rate = {args.rcga_mutation_rate}',
        '',
        f'sca.a = {args.sca_a}',
    ]
    config_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='Runner for single optimization experiment in C++'
    )

    parser.add_argument('--session-dir', type=str)
    parser.add_argument('--run-id', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--problem', type=str, default='ihcp')
    parser.add_argument('--noise-level', dest='noise_level', type=float, default=0.03)
    parser.add_argument('--optimizer', type=str, default='aoa', choices=OPTIMIZERS)
    parser.add_argument('--n-runs', type=int, default=1)
    
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='Silence C++ backend standard output logging'
    )

    parser.add_argument(
        '--n-solutions', '--n-pop', dest='n_solutions', type=int, default=30
    )
    parser.add_argument('--n-iter', dest='n_iter', type=int, default=30)

    parser.add_argument('--aco-locality', dest='aco_locality', type=float, default=0.3)
    parser.add_argument(
        '--aco-evaporation', dest='aco_evaporation', type=float, default=0.85
    )

    parser.add_argument('--aoa-c1', dest='aoa_c1', type=float, default=2.0)
    parser.add_argument('--aoa-c2', dest='aoa_c2', type=float, default=6.0)
    parser.add_argument('--aoa-c3', dest='aoa_c3', type=float, default=2.0)
    parser.add_argument('--aoa-c4', dest='aoa_c4', type=float, default=0.5)

    parser.add_argument('--cs-pa', dest='cs_pa', type=float, default=0.25)

    parser.add_argument('--pso-inertia', dest='pso_inertia', type=float, default=0.7)
    parser.add_argument('--pso-c1', dest='pso_c1', type=float, default=1.5)
    parser.add_argument('--pso-c2', dest='pso_c2', type=float, default=1.5)

    parser.add_argument(
        '--rcga-crossover-rate', dest='rcga_crossover_rate', type=float, default=0.8
    )
    parser.add_argument(
        '--rcga-mutation-rate', dest='rcga_mutation_rate', type=float, default=0.1
    )

    parser.add_argument('--sca-a', dest='sca_a', type=float, default=2.0)

    args = parser.parse_args()

    session_dir = (
        Path(args.session_dir).resolve() if args.session_dir else generate_session_dir()
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id if args.run_id else generate_run_id()
    log_file_path = session_dir / f'run_{run_id}.log'
    config_file_path = session_dir / f'run_{run_id}.cfg'

    write_config(config_file_path, session_dir, run_id, args)

    cmd = [str(EXE_PATH), str(config_file_path)]

    try:
        start_time = time.perf_counter()
        
        if args.quiet:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        else:
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                result = subprocess.run(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True
                )
                
        end_time = time.perf_counter()
        print_info(f'Experiment took {end_time - start_time:.2f}s')

        if result.returncode != 0:
            if args.quiet:
                with open(log_file_path, 'w', encoding='utf-8') as log_file:
                    log_file.write(result.stdout)
                    
            print_error(
                f'[{args.optimizer.upper()}] FAILED run_{run_id} (code {result.returncode})! Details: {log_file_path.name}',
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    except FileNotFoundError:
        print_error(f'Executable not found at path: {EXE_PATH}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()