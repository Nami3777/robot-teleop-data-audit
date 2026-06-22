import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_NAME = "lerobot/pusht"


@dataclass
class LoadedTeleopData:
    name: str
    frame: pd.DataFrame
    source: str
    notes: list[str]


def reports_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect public robot teleoperation data.")
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use a tiny local synthetic dataset for learning and offline verification.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Maximum rows to keep for quick local inspection.",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=10,
        help=(
            "Minimum frames for a valid episode. "
            "Set to your task's minimum expected demonstration length "
            "(e.g. 40 for pick-and-place, 80 for manipulation sequences). "
            "Default of 10 is a conservative floor."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="audit_summary",
        help=(
            "Prefix for generated report files. "
            "Use this to keep public-data and synthetic verification runs separate."
        ),
    )
    return parser.parse_args()


def load_public_or_synthetic(use_synthetic: bool, max_rows: int = 5000) -> LoadedTeleopData:
    if use_synthetic:
        return load_synthetic_data(max_rows=max_rows)

    try:
        from datasets import load_dataset
    except ImportError:
        data = load_synthetic_data(max_rows=max_rows)
        data.notes.append("datasets package is not installed; used synthetic fallback.")
        return data

    notes: list[str] = []
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        table = dataset.select(range(min(max_rows, len(dataset)))).to_pandas()
        notes.append(f"Loaded {DATASET_NAME} from Hugging Face.")
        notes.append(f"Rows kept for audit: {len(table)} of {len(dataset)}.")
        return LoadedTeleopData(DATASET_NAME, table, "huggingface", notes)
    except Exception as exc:  # noqa: BLE001 - fallback keeps the learning workflow usable.
        data = load_synthetic_data(max_rows=max_rows)
        data.notes.append(f"Could not load {DATASET_NAME}; used synthetic fallback. Error: {exc}")
        return data


def load_synthetic_data(max_rows: int = 5000) -> LoadedTeleopData:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed=7)
    episode_lengths = [8, 24, 30, 4, 28]
    for episode_index, length in enumerate(episode_lengths):
        action = np.array([0.0, 0.0], dtype=float)
        for frame_index in range(length):
            if not (episode_index == 2 and 10 <= frame_index <= 15):
                action = action + rng.normal(0, 0.08, size=2)
            timestamp = frame_index * 0.1
            if episode_index == 1 and frame_index > 15:
                timestamp += 0.45
            rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "observation.state": rng.normal(0, 1, size=4).round(4).tolist(),
                    "action": action.round(4).tolist(),
                    "task": "push target to goal",
                }
            )

    frame = pd.DataFrame(rows).head(max_rows)
    notes = [
        "Loaded synthetic fallback data.",
        "Run without --use-synthetic to audit public LeRobot data (downloaded on first run).",
    ]
    return LoadedTeleopData("synthetic/teleop_debug", frame, "synthetic", notes)


def detect_schema(frame: pd.DataFrame) -> dict[str, Any]:
    columns = list(frame.columns)
    column_set = set(columns)
    episode_col = next(
        (c for c in ("episode_index", "episode_id", "episode", "index.episode") if c in column_set),
        None,
    )
    timestamp_col = next(
        (c for c in ("timestamp", "timestamps", "time", "frame.timestamp") if c in column_set),
        None,
    )
    action_col = next((c for c in ("action", "actions") if c in column_set), None)
    task_col = next((c for c in ("task", "task_index", "task_label") if c in column_set), None)
    observation_cols = [
        c
        for c in columns
        if any(fragment in c.lower() for fragment in ("observation", "obs", "state", "image"))
    ]

    return {
        "columns": columns,
        "episode_col": episode_col,
        "timestamp_col": timestamp_col,
        "action_col": action_col,
        "task_col": task_col,
        "observation_cols": observation_cols,
    }


def values_to_numeric_matrix(values: pd.Series) -> np.ndarray:
    converted: list[np.ndarray] = []
    for value in values:
        if value is None:
            converted.append(np.array([], dtype=float))
            continue
        if isinstance(value, np.ndarray):
            array = value.astype(float, copy=False).reshape(-1)
        elif isinstance(value, (list, tuple)):
            array = np.array(value, dtype=float).reshape(-1)
        else:
            try:
                array = np.array([float(value)], dtype=float)
            except (TypeError, ValueError):
                array = np.array([], dtype=float)
        converted.append(array)

    width = max((len(item) for item in converted), default=0)
    if width == 0:
        return np.empty((len(converted), 0))

    matrix = np.full((len(converted), width), np.nan, dtype=float)
    for index, array in enumerate(converted):
        matrix[index, : len(array)] = array
    return matrix


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)
