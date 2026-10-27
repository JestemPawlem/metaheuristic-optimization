import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from common import print_error, print_info


def get_latest_session_dir(results_dir: Path) -> Path:
    sessions = sorted([d for d in results_dir.glob('session_*') if d.is_dir()])
    if not sessions:
        raise FileNotFoundError(f'No session directories found in: {results_dir}')
    return sessions[-1]


def setup_plot_style():
    sns.set_theme(style='whitegrid')
    plt.rcParams.update(
        {
            'font.sans-serif': 'DejaVu Sans',
            'axes.edgecolor': '#cccccc',
            'axes.linewidth': 0.8,
        }
    )


def generate_plots(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)

    required_cols = {'algorithm', 'fitness', 'mean_error', 'n_evaluations'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f'CSV missing required columns: {required_cols}. Found: {set(df.columns)}'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df,
        x='n_evaluations',
        y='fitness',
        hue='algorithm',
        marker='o',
        ax=ax,
    )
    ax.set_yscale('log')
    ax.set_title(
        'Best Fitness vs Fitness Evaluations',
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlabel('Evaluations', fontsize=10)
    ax.set_ylabel('Best Fitness', fontsize=10)
    fig.tight_layout()

    fitness_plot = output_dir / 'fitness_vs_evaluations.png'
    fig.savefig(fitness_plot)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.lineplot(
        data=df,
        x='n_evaluations',
        y='mean_error',
        hue='algorithm',
        marker='s',
        ax=ax,
    )
    ax.set_yscale('log')
    ax.set_title(
        'Mean Error vs Fitness Evaluations',
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlabel('Evaluations', fontsize=10)
    ax.set_ylabel('Mean Error', fontsize=10)
    fig.tight_layout()

    error_plot = output_dir / 'mean_error_vs_evaluations.png'
    fig.savefig(error_plot)
    plt.close(fig)

    print_info(f'Generated artifacts in: {output_dir.resolve()}')
    print_info(f' - {fitness_plot.name}')
    print_info(f' - {error_plot.name}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark metric plots for IHCP optimization'
    )
    parser.add_argument(
        '--session-dir',
        type=str,
        default=None,
        help='Target session directory (default: latest session)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to store generated plots (default: artifacts/ inside session directory)',
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / 'results'

    try:
        session_dir = (
            Path(args.session_dir).resolve()
            if args.session_dir
            else get_latest_session_dir(results_dir)
        )
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    csv_file = session_dir / 'aggregated_results.csv'
    if not csv_file.exists():
        print_error(f'File not found: {csv_file}')
        sys.exit(1)

    artifacts_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else session_dir / 'artifacts'
    )

    generate_plots(csv_file, artifacts_dir)


if __name__ == '__main__':
    main()