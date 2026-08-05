"""Evaluate the trained steering model and create report-ready results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from preprocessing import load_rgb_image, preprocess_image

COLUMNS = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
TURN_RANGES = (
    ("Near straight", 0.0, 0.05),
    ("Small turn", 0.05, 0.20),
    ("Medium turn", 0.20, 0.40),
    ("Sharp turn", 0.40, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model.h5 on the original validation split or a held-out log."
    )
    parser.add_argument("--model", type=Path, default=Path("model.h5"))
    parser.add_argument(
        "--csv", type=Path, required=True, help="Path to driving_log.csv"
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Folder that contains IMG (default: the CSV folder)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--split",
        choices=("validation", "all"),
        default="validation",
        help="Recreate the notebook validation split, or evaluate every CSV row",
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--zero-drop-fraction",
        type=float,
        default=0.8,
        help="Fraction of exact 0.0 steering rows to remove, as in training",
    )
    parser.add_argument("--sequence-length", type=int, default=500)
    parser.add_argument("--worst-count", type=int, default=12)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0 <= args.zero_drop_fraction < 1:
        parser.error("--zero-drop-fraction must be in the range [0, 1)")
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size must be in the range (0, 1)")
    return args


def read_driving_log(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path, names=COLUMNS, header=None)
    data["Steering"] = pd.to_numeric(data["Steering"], errors="coerce")
    data = data.dropna(subset=["Center", "Steering"]).copy()
    data["Center"] = data["Center"].astype(str).str.strip()
    data["_source_index"] = np.arange(len(data))
    return data


def select_evaluation_rows(
    data: pd.DataFrame,
    split: str,
    zero_drop_fraction: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    zero_rows = data[data["Steering"] == 0.0]
    drop_indices = zero_rows.sample(frac=zero_drop_fraction, random_state=seed).index
    balanced = data.drop(drop_indices)

    if split == "validation":
        _, selected = train_test_split(
            balanced,
            test_size=validation_size,
            random_state=seed,
        )
    else:
        selected = balanced

    return selected.sort_values("_source_index").reset_index(drop=True), len(
        drop_indices
    )


def resolve_image_path(raw_path: str, csv_path: Path, image_root: Path) -> Path | None:
    normalized = raw_path.replace("\\", "/")
    filename = PurePosixPath(normalized).name
    relative = Path(*PurePosixPath(normalized).parts)
    candidates = [
        Path(raw_path).expanduser(),
        image_root / relative,
        image_root / "IMG" / filename,
        csv_path.parent / relative,
        csv_path.parent / "IMG" / filename,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_samples(
    data: pd.DataFrame, csv_path: Path, image_root: Path
) -> tuple[pd.DataFrame, int]:
    resolved_paths: list[str | None] = []
    for raw_path in data["Center"]:
        resolved = resolve_image_path(raw_path, csv_path, image_root)
        resolved_paths.append(str(resolved) if resolved else None)

    resolved_data = data.copy()
    resolved_data["image_path"] = resolved_paths
    missing_count = int(resolved_data["image_path"].isna().sum())
    resolved_data = resolved_data.dropna(subset=["image_path"]).reset_index(drop=True)
    if resolved_data.empty:
        example = data.iloc[0]["Center"] if not data.empty else "<no rows>"
        raise FileNotFoundError(
            "No center-camera images were found. "
            f"Check --image-root. First CSV path: {example}"
        )
    return resolved_data, missing_count


def predict(model: Any, paths: list[str], batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    total_batches = (len(paths) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(paths), batch_size), start=1):
        batch_paths = paths[start : start + batch_size]
        images = np.stack(
            [preprocess_image(load_rgb_image(path)) for path in batch_paths]
        )
        batch_predictions = np.asarray(model(images, training=False)).reshape(-1)
        predictions.append(batch_predictions)
        print(f"Predicting batch {batch_number}/{total_batches}", end="\r", flush=True)

    print()
    return np.concatenate(predictions).astype(float)


def basic_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def label_turn_range(angle: float) -> str:
    magnitude = abs(angle)
    for name, lower, upper in TURN_RANGES:
        if lower <= magnitude < upper:
            return name
    raise AssertionError("TURN_RANGES must cover all steering values")


def metrics_by_turn_range(
    results: pd.DataFrame,
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for name, _, _ in TURN_RANGES:
        group = results[results["turn_range"] == name]
        if group.empty:
            output[name] = {"count": 0, "mae": None, "rmse": None}
            continue
        output[name] = {
            "count": int(len(group)),
            "mae": float(group["absolute_error"].mean()),
            "rmse": float(np.sqrt(np.mean(np.square(group["error"])))),
        }
    return output


def save_actual_vs_predicted(results: pd.DataFrame, output_dir: Path) -> None:
    low = float(min(results["actual"].min(), results["predicted"].min()))
    high = float(max(results["actual"].max(), results["predicted"].max()))
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(results["actual"], results["predicted"], alpha=0.3, s=14)
    axis.plot([low, high], [low, high], "r--", label="Ideal prediction")
    axis.set(title="Actual vs Predicted Steering", xlabel="Actual", ylabel="Predicted")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def save_error_histogram(results: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(results["error"], bins=60, color="steelblue", edgecolor="white")
    axis.axvline(0, color="red", linestyle="--", linewidth=1)
    axis.set(
        title="Steering Prediction Error",
        xlabel="Predicted minus actual steering",
        ylabel="Samples",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "error_histogram.png", dpi=160)
    plt.close(fig)


def save_steering_sequence(
    results: pd.DataFrame, output_dir: Path, sequence_length: int
) -> None:
    sequence = results.head(sequence_length)
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(sequence["actual"].to_numpy(), label="Actual", linewidth=1.4)
    axis.plot(sequence["predicted"].to_numpy(), label="Predicted", linewidth=1.2)
    axis.set(
        title=f"Steering Sequence (first {len(sequence)} evaluated frames)",
        xlabel="Frame order",
        ylabel="Steering angle",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "steering_sequence.png", dpi=160)
    plt.close(fig)


def save_range_performance(
    range_metrics: dict[str, dict[str, float | int | None]], output_dir: Path
) -> None:
    labels = list(range_metrics)
    values = [range_metrics[label]["mae"] or 0.0 for label in labels]
    counts = [int(range_metrics[label]["count"] or 0) for label in labels]
    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f2cf5b", "#e45756"])
    axis.set(title="Error by Steering Range", ylabel="Mean absolute error")
    axis.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_turn_range.png", dpi=160)
    plt.close(fig)


def save_worst_predictions(results: pd.DataFrame, output_dir: Path, count: int) -> None:
    worst = results.nlargest(min(count, len(results)), "absolute_error")
    columns = 3
    rows = int(np.ceil(len(worst) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.8 * rows))
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, (_, row) in zip(axes_array, worst.iterrows()):
        axis.imshow(load_rgb_image(row["image_path"]))
        axis.set_title(
            f"Actual {row['actual']:+.3f} | Pred {row['predicted']:+.3f}\n"
            f"Absolute error {row['absolute_error']:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(worst) :]:
        axis.axis("off")

    fig.suptitle("Largest Steering Prediction Errors", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "worst_predictions.png", dpi=160)
    plt.close(fig)


def write_text_summary(metrics: dict[str, Any], output_path: Path) -> None:
    overall = metrics["model"]
    baseline = metrics["zero_steering_baseline"]
    lines = [
        "DPS920 Group 17 - Offline Model Evaluation",
        "=" * 47,
        f"Evaluation split: {metrics['data']['split']}",
        f"Evaluated samples: {metrics['data']['evaluated_samples']}",
        f"Missing image rows skipped: {metrics['data']['missing_images']}",
        f"Exact-zero rows removed: {metrics['data']['zero_rows_removed']}",
        "",
        f"Model MAE:  {overall['mae']:.6f}",
        f"Model MSE:  {overall['mse']:.6f}",
        f"Model RMSE: {overall['rmse']:.6f}",
        f"Model R2:   {overall['r2']:.6f}",
        "",
        f"Zero baseline MAE:  {baseline['mae']:.6f}",
        f"Zero baseline MSE:  {baseline['mse']:.6f}",
        f"Zero baseline RMSE: {baseline['rmse']:.6f}",
        f"RMSE improvement over zero baseline: {metrics['rmse_improvement_percent']:.2f}%",
        "",
        "Error by steering range:",
    ]
    for name, values in metrics["by_turn_range"].items():
        if values["count"]:
            lines.append(
                f"- {name}: n={values['count']}, "
                f"MAE={values['mae']:.6f}, RMSE={values['rmse']:.6f}"
            )
        else:
            lines.append(f"- {name}: n=0")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    image_root = (args.image_root or csv_path.parent).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_driving_log(csv_path)
    selected, zero_rows_removed = select_evaluation_rows(
        data,
        split=args.split,
        zero_drop_fraction=args.zero_drop_fraction,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    selected, missing_count = resolve_samples(selected, csv_path, image_root)

    print(f"Loading model: {args.model}")
    model = load_model(args.model, compile=False)
    actual = selected["Steering"].to_numpy(dtype=float)
    predicted = predict(model, selected["image_path"].tolist(), args.batch_size)

    results = pd.DataFrame(
        {
            "source_index": selected["_source_index"].to_numpy(),
            "csv_image_path": selected["Center"].to_numpy(),
            "image_path": selected["image_path"].to_numpy(),
            "actual": actual,
            "predicted": predicted,
        }
    )
    results["error"] = results["predicted"] - results["actual"]
    results["absolute_error"] = results["error"].abs()
    results["turn_range"] = results["actual"].map(label_turn_range)

    model_metrics = basic_metrics(actual, predicted)
    baseline_metrics = basic_metrics(actual, np.zeros_like(actual))
    improvement = (
        100.0
        * (baseline_metrics["rmse"] - model_metrics["rmse"])
        / baseline_metrics["rmse"]
        if baseline_metrics["rmse"]
        else 0.0
    )
    range_metrics = metrics_by_turn_range(results)
    metrics: dict[str, Any] = {
        "data": {
            "csv": str(args.csv),
            "image_root": str(args.image_root or args.csv.parent),
            "split": args.split,
            "source_rows": int(len(data)),
            "zero_rows_removed": int(zero_rows_removed),
            "missing_images": missing_count,
            "evaluated_samples": int(len(results)),
            "seed": args.seed,
            "validation_size": args.validation_size
            if args.split == "validation"
            else None,
        },
        "model": model_metrics,
        "zero_steering_baseline": baseline_metrics,
        "rmse_improvement_percent": float(improvement),
        "by_turn_range": range_metrics,
    }

    exported_results = results.drop(columns="image_path").rename(
        columns={"csv_image_path": "image_path"}
    )
    exported_results.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_text_summary(metrics, output_dir / "metrics.txt")
    save_actual_vs_predicted(results, output_dir)
    save_error_histogram(results, output_dir)
    save_steering_sequence(results, output_dir, args.sequence_length)
    save_range_performance(range_metrics, output_dir)
    save_worst_predictions(results, output_dir, args.worst_count)

    print((output_dir / "metrics.txt").read_text(encoding="utf-8"))
    print(f"Saved evaluation outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
