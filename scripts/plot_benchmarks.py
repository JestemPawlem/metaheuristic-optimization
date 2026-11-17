import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from common import print_error, print_info


def get_latest_session_dir(results_dir: Path) -> Path:
    sessions = sorted([d for d in results_dir.glob("session_*") if d.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No session directories found in: {results_dir}")
    return sessions[-1]


def read_problem_from_cfg(cfg_path: Path) -> str:
    if not cfg_path.exists():
        return "unknown"

    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, val = (part.strip() for part in line.split("=", 1))
            if key == "problem":
                return val

    return "unknown"


def load_and_aggregate_data(session_dir: Path) -> pd.DataFrame:
    csv_files = sorted(session_dir.glob("run_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No run_*.csv files found in: {session_dir}")

    frames = []
    for f in csv_files:
        part = pd.read_csv(f)
        if "problem" not in part.columns:
            part["problem"] = read_problem_from_cfg(f.with_suffix(".cfg"))
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    aggregated_path = session_dir / "aggregated_results.csv"
    df.to_csv(aggregated_path, index=False)
    print_info(
        f"Aggregated {len(df)} rows from {len(csv_files)} files -> {aggregated_path.name}"
    )

    return df


def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.sans-serif": "DejaVu Sans",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
        }
    )


def generate_table_image(stats_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, len(stats_df) * 0.35 + 1.5), dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    for (i, _), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")

    plt.title(
        "Benchmark Final Statistics",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_plots(df: pd.DataFrame, output_dir: Path):
    required_cols = {
        "problem",
        "algorithm",
        "fitness",
        "n_evaluations",
        "n_solutions",
        "n_iterations",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"DataFrame missing required columns: {required_cols - set(df.columns)}"
        )

    df["n_evaluations"] = (
        df.groupby(["problem", "algorithm", "n_solutions", "n_iterations"])[
            "n_evaluations"
        ]
        .transform("mean")
        .round()
        .astype(int)
    )

    df["fitness"] = df["fitness"].clip(lower=1e-6)

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    generated_files = []

    for problem_name, problem_df in df.groupby("problem"):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sns.lineplot(
            data=problem_df,
            x="n_evaluations",
            y="fitness",
            hue="algorithm",
            marker="o",
            ax=ax,
            estimator="median",
            errorbar=("pi", 50),
        )
        ax.set_yscale("log")
        ax.set_title(
            f"Best Fitness vs Fitness Evaluations ({problem_name})",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Evaluations", fontsize=10)
        ax.set_ylabel("Best Fitness", fontsize=10)
        fig.tight_layout()

        plot_path = output_dir / f"{problem_name}_fitness_vs_evaluations.png"
        fig.savefig(plot_path)
        plt.close(fig)
        generated_files.append(plot_path.name)

    final_results = df[df["n_evaluations"] == df["n_evaluations"].max()]

    stats_table = (
        final_results.groupby(["problem", "algorithm"])["fitness"]
        .agg(
            f_min="min",
            f_max="max",
            f_mean="mean",
            f_median="median",
            f_std="std",
        )
        .reset_index()
    )

    cols = ["f_min", "f_max", "f_mean", "f_median", "f_std"]
    for col in cols:
        stats_table[col] = stats_table[col].apply(lambda x: f"{x:.4f}")

    csv_path = output_dir / "clean_statistics.csv"
    stats_table.to_csv(csv_path, index=False)
    generated_files.append(csv_path.name)

    table_img_path = output_dir / "final_statistics_table.png"
    generate_table_image(stats_table, table_img_path)
    generated_files.append(table_img_path.name)

    print_info(f"Generated artifacts in: {output_dir.resolve()}")
    for name in generated_files:
        print_info(f" - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate run results and generate benchmark function plots"
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Target session directory (default: latest session)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store generated plots (default: artifacts/ inside session directory)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"

    try:
        session_dir = (
            Path(args.session_dir).resolve()
            if args.session_dir
            else get_latest_session_dir(results_dir)
        )
        df = load_and_aggregate_data(session_dir)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    artifacts_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else session_dir / "artifacts"
    )

    generate_plots(df, artifacts_dir)


if __name__ == "__main__":
    main()