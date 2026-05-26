"""
Run evaluate.py across multiple random data selections and aggregate results.
Each run creates a fresh selected dataset with a random selection seed, 
runs evaluate.py, and then aggregates the model comparison CSV files across 
those random selections.
"""

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import PROJECT_ROOT


DEFAULT_CONFIG = "config/experiment_config.json"
DEFAULT_OUTPUT_DIR = "results/random_selection_averaged"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluate.py multiple times with random data selections and average the results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to evaluate.py JSON config.")
    parser.add_argument("--selection-runs", type=int, default=5, help="Number of random selections.")
    parser.add_argument(
        "--inner-num-runs",
        type=int,
        default=1,
        help="Number of training runs inside each evaluate.py call.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for all outputs.")
    parser.add_argument(
        "--seed-min",
        type=int,
        default=1,
        help="Minimum generated selection seed.",
    )
    parser.add_argument(
        "--seed-max",
        type=int,
        default=2_147_483_647,
        help="Maximum generated selection seed.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run evaluate.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print evaluate.py commands without running them.",
    )
    args, evaluate_args = parser.parse_known_args()
    return args, evaluate_args


def main():
    args, evaluate_args = parse_args()
    validate_passthrough_args(evaluate_args)
    if args.selection_runs < 1:
        raise ValueError("--selection-runs must be at least 1")
    if args.inner_num_runs < 1:
        raise ValueError("--inner-num-runs must be at least 1")

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_seeds = generate_unique_seeds(
        count=args.selection_runs,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "config": str(resolve_project_path(args.config)),
        "selection_runs": args.selection_runs,
        "inner_num_runs": args.inner_num_runs,
        "selection_seeds": selection_seeds,
        "evaluate_args": evaluate_args,
        "runs": [],
    }

    print("=" * 80)
    print("RANDOM-SELECTION EVALUATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Selection runs: {args.selection_runs}")
    print(f"Inner evaluate.py num-runs: {args.inner_num_runs}")
    print(f"Selection seeds: {selection_seeds}")
    print("=" * 80)

    for index, selection_seed in enumerate(selection_seeds, 1):
        run_dir = output_dir / f"selection_run_{index:02d}"
        selected_data_dir = run_dir / "training_data"
        evaluate_results_dir = run_dir / "evaluate_results"
        selected_data_dir.mkdir(parents=True, exist_ok=True)
        evaluate_results_dir.mkdir(parents=True, exist_ok=True)

        command = build_evaluate_command(
            python_executable=args.python,
            config_path=resolve_project_path(args.config),
            selected_data_dir=selected_data_dir,
            results_dir=evaluate_results_dir,
            selection_seed=selection_seed,
            inner_num_runs=args.inner_num_runs,
            extra_args=evaluate_args,
        )

        run_record = {
            "run_index": index,
            "selection_seed": selection_seed,
            "selected_data_dir": str(selected_data_dir),
            "results_dir": str(evaluate_results_dir),
            "command": command,
            "status": "pending",
        }
        manifest["runs"].append(run_record)

        print("\n" + "-" * 80)
        print(f"Selection run {index}/{args.selection_runs}")
        print(f"selection_seed={selection_seed}")
        print(" ".join(quote_command_part(part) for part in command))

        if args.dry_run:
            run_record["status"] = "dry_run"
            continue

        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        run_record["returncode"] = completed.returncode
        if completed.returncode == 0:
            run_record["status"] = "success"
        else:
            run_record["status"] = "failed"
            write_manifest(output_dir, manifest)
            raise RuntimeError(f"evaluate.py failed on selection run {index} with code {completed.returncode}")

    write_manifest(output_dir, manifest)

    if args.dry_run:
        print(f"\nDry run complete. Manifest written to: {output_dir / 'run_manifest.json'}")
        return

    aggregate_results(output_dir, manifest)


def build_evaluate_command(
    python_executable,
    config_path,
    selected_data_dir,
    results_dir,
    selection_seed,
    inner_num_runs,
    extra_args,
):
    return [
        python_executable,
        str(PROJECT_ROOT / "src" / "evaluate.py"),
        "--config",
        str(config_path),
        "--selection-seed",
        str(selection_seed),
        "--selected-data-dir",
        str(selected_data_dir),
        "--results-dir",
        str(results_dir),
        "--num-runs",
        str(inner_num_runs),
        *extra_args,
    ]


def aggregate_results(output_dir, manifest):
    frames = []
    for run in manifest["runs"]:
        if run.get("status") != "success":
            continue

        comparison_file = Path(run["results_dir"]) / "model_comparison_summary.csv"
        if not comparison_file.exists():
            print(f"[WARNING] Missing comparison file: {comparison_file}")
            continue

        frame = pd.read_csv(comparison_file)
        frame.insert(0, "selection_seed", run["selection_seed"])
        frame.insert(0, "selection_run", run["run_index"])
        frames.append(frame)

    if not frames:
        raise RuntimeError("No successful model_comparison_summary.csv files were found.")

    detailed = pd.concat(frames, ignore_index=True)
    detailed_path = output_dir / "per_selection_model_comparison.csv"
    detailed.to_csv(detailed_path, index=False)

    metric_columns = [
        column for column in detailed.columns
        if column not in {"selection_run", "selection_seed", "Approach", "Model"}
    ]
    for column in metric_columns:
        detailed[column] = pd.to_numeric(detailed[column], errors="coerce")

    grouped = detailed.groupby(["Approach", "Model"], dropna=False)
    summary_rows = []
    for (approach, model), group in grouped:
        row = {
            "Approach": approach,
            "Model": model,
            "selection_runs": int(group["selection_run"].nunique()),
        }
        for metric in metric_columns:
            values = group[metric].dropna()
            if values.empty:
                continue
            row[f"{metric}_selection_mean"] = values.mean()
            row[f"{metric}_selection_std"] = values.std(ddof=0)
            row[f"{metric}_selection_min"] = values.min()
            row[f"{metric}_selection_max"] = values.max()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if "accuracy_mean_selection_mean" in summary.columns:
        summary = summary.sort_values("accuracy_mean_selection_mean", ascending=False)

    summary_path = output_dir / "random_selection_averaged_summary.csv"
    summary.to_csv(summary_path, index=False)

    json_path = output_dir / "random_selection_averaged_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "manifest": manifest,
                "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
            },
            file,
            indent=2,
        )

    report_path = output_dir / "random_selection_averaged_results.txt"
    write_text_report(report_path, summary, manifest)

    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"Detailed per-selection CSV: {detailed_path}")
    print(f"Averaged summary CSV: {summary_path}")
    print(f"Averaged summary JSON: {json_path}")
    print(f"Text report: {report_path}")


def write_text_report(report_path, summary, manifest):
    with report_path.open("w", encoding="utf-8") as file:
        write = lambda text="": file.write(text + "\n")
        write("=" * 80)
        write("RANDOM-SELECTION AVERAGED RESULTS")
        write("=" * 80)
        write(f"Generated: {datetime.now().isoformat()}")
        write(f"Selection runs: {manifest['selection_runs']}")
        write(f"Inner evaluate.py num-runs: {manifest['inner_num_runs']}")
        write(f"Selection seeds: {manifest['selection_seeds']}")
        write("")

        if summary.empty:
            write("No results available.")
            return

        accuracy_column = "accuracy_mean_selection_mean"
        accuracy_std_column = "accuracy_mean_selection_std"
        f1_column = "f1_score_macro_mean_selection_mean"
        f1_std_column = "f1_score_macro_mean_selection_std"

        for _, row in summary.iterrows():
            write(f"{row['Approach']} / {row['Model']}")
            write(f"  selection_runs: {int(row['selection_runs'])}")
            if accuracy_column in row and pd.notna(row[accuracy_column]):
                write(
                    f"  accuracy: {row[accuracy_column]:.2f}%"
                    f" +/- {row.get(accuracy_std_column, 0):.2f}%"
                )
            if f1_column in row and pd.notna(row[f1_column]):
                write(
                    f"  macro_f1: {row[f1_column]:.2f}%"
                    f" +/- {row.get(f1_std_column, 0):.2f}%"
                )
            write("")


def generate_unique_seeds(count, seed_min, seed_max):
    if seed_max < seed_min:
        raise ValueError("--seed-max must be greater than or equal to --seed-min")
    if seed_max - seed_min + 1 < count:
        raise ValueError("Seed range is too small for the requested number of runs")

    generator = random.SystemRandom()
    seen = set()
    seeds = []
    while len(seeds) < count:
        seed = generator.randint(seed_min, seed_max)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def validate_passthrough_args(evaluate_args):
    controlled_args = {
        "--selection-seed",
        "--selected-data-dir",
        "--results-dir",
        "--num-runs",
        "--skip-selection",
    }
    used_controlled_args = [arg for arg in evaluate_args if arg in controlled_args]
    if used_controlled_args:
        joined = ", ".join(used_controlled_args)
        raise ValueError(
            "These evaluate.py arguments are controlled by random_selection_evaluate.py "
            f"and cannot be passed through: {joined}"
        )


def resolve_project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_manifest(output_dir, manifest):
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def quote_command_part(part):
    text = str(part)
    if " " in text:
        return f'"{text}"'
    return text


if __name__ == "__main__":
    main()
