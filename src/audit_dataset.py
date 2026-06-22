from typing import Any

import numpy as np
import pandas as pd

from teleop_data import (
    detect_schema,
    load_public_or_synthetic,
    markdown_table,
    parse_args,
    reports_dir,
    values_to_numeric_matrix,
)


LONG_EPISODE_FACTOR = 2.0
STUCK_ACTION_DELTA = 1e-6
STUCK_ACTION_RUN_LENGTH = 5
TIMESTAMP_GAP_FACTOR = 3.0

FLAG_DESCRIPTIONS: dict[str, str] = {
    "short_episode": (
        "Possible aborted attempt or incomplete task. "
        "Verify against minimum task SOP before including in training."
    ),
    "long_episode": (
        "Unusual execution time. "
        "Check for operator uncertainty, task interruption, or inconsistent segmentation."
    ),
    "repeated_or_stuck_action": (
        "Operator hesitation indicator - near-stationary action sequence detected. "
        "These segments add low-information frames that a policy can overfit on."
    ),
    "timestamp_gap": (
        "Logging interruption or sensor handoff gap detected. "
        "Data integrity in this window is uncertain; prefer re-collection."
    ),
}


def count_missing(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"column": column, "missing_count": int(missing)}
        for column, missing in frame.isna().sum().items()
        if int(missing)
    ]


def longest_stuck_action_run(actions: np.ndarray) -> int:
    if len(actions) < 2 or actions.shape[1] == 0:
        return 0
    deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    longest = 0
    current = 0
    for delta in deltas:
        if np.isfinite(delta) and delta <= STUCK_ACTION_DELTA:
            current += 1
            longest = max(longest, current + 1)
        else:
            current = 0
    return longest


def timestamp_gap_count(values: pd.Series) -> int:
    timestamps = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(timestamps) < 3:
        return 0
    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 0
    median_gap = float(np.median(positive_diffs))
    if median_gap <= 0:
        return 0
    return int(np.sum(diffs > median_gap * TIMESTAMP_GAP_FACTOR))


def status_from_flags(flags: list[str]) -> str:
    if not flags:
        return "pass"
    return "flag" if {"short_episode", "timestamp_gap"} & set(flags) else "review"


def audit(
    frame: pd.DataFrame,
    min_episode_frames: int = 10,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    schema = detect_schema(frame)
    episode_col = schema["episode_col"]
    timestamp_col = schema["timestamp_col"]
    action_col = schema["action_col"]

    episodes = list(frame.groupby(episode_col, sort=True)) if episode_col else [("unknown_episode", frame)]
    lengths = [len(episode) for _episode_id, episode in episodes]

    median_length = float(np.median(lengths)) if lengths else 0.0
    long_threshold = max(min_episode_frames + 1, int(median_length * LONG_EPISODE_FACTOR))

    episode_rows: list[dict[str, Any]] = []
    for episode_id, episode in episodes:
        length = len(episode)

        action_run = 0
        if action_col:
            action_run = longest_stuck_action_run(values_to_numeric_matrix(episode[action_col]))

        gap_count = 0
        if timestamp_col:
            gap_count = timestamp_gap_count(episode[timestamp_col])

        flags: list[str] = []
        if length < min_episode_frames:
            flags.append("short_episode")
        if length > long_threshold:
            flags.append("long_episode")
        if action_run >= STUCK_ACTION_RUN_LENGTH:
            flags.append("repeated_or_stuck_action")
        if gap_count:
            flags.append("timestamp_gap")

        episode_rows.append(
            {
                "episode_id": episode_id,
                "frame_count": length,
                "timestamp_gap_count": gap_count,
                "longest_stuck_action_run": action_run,
                "status": status_from_flags(flags),
                "flags": ",".join(flags) if flags else "none",
            }
        )

    summary = {
        "episode_count": len(episode_rows),
        "frame_count": len(frame),
        "median_episode_length": median_length,
        "min_episode_frames": min_episode_frames,
        "long_episode_threshold": long_threshold,
        "episode_col": episode_col,
        "timestamp_col": timestamp_col,
        "action_col": action_col,
    }

    return pd.DataFrame(episode_rows), count_missing(frame), summary


def session_pattern_analysis(episode_audit: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Distributes flagged episodes across session quartiles (by episode order).
    Clustering in Q3 or Q4 suggests session degradation - operator fatigue,
    equipment drift, or a late-session interruption - rather than random noise.
    """
    if len(episode_audit) < 4:
        return []
    indexed = episode_audit.reset_index(drop=True)
    indexed["_quartile"] = pd.qcut(
        indexed.index,
        q=4,
        labels=["Q1 (earliest)", "Q2", "Q3", "Q4 (latest)"],
    )
    rows = []
    for label, group in indexed.groupby("_quartile", observed=True):
        flagged = int((group["flags"] != "none").sum())
        rows.append(
            {
                "session_quartile": str(label),
                "episodes": len(group),
                "flagged": flagged,
                "flag_rate": f"{flagged / len(group):.0%}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    loaded = load_public_or_synthetic(args.use_synthetic, args.max_rows)
    episode_audit, missing_rows, summary = audit(
        loaded.frame,
        min_episode_frames=args.min_episode_frames,
    )

    output_dir = reports_dir()
    output_prefix = args.output_prefix.strip() or "audit_summary"
    csv_path = output_dir / f"{output_prefix}.csv"
    md_path = output_dir / f"{output_prefix}.md"
    episode_audit.to_csv(csv_path, index=False)

    pass_count = int((episode_audit["status"] == "pass").sum())
    review_count = int((episode_audit["status"] == "review").sum())
    flag_count = int((episode_audit["status"] == "flag").sum())

    flagged_rows = episode_audit[episode_audit["flags"] != "none"].to_dict(orient="records")
    session_rows = session_pattern_analysis(episode_audit)

    flag_legend = [
        f"- **{flag}**: {desc}" for flag, desc in FLAG_DESCRIPTIONS.items()
    ]

    session_note = (
        "Flag distribution across the collection session by episode order. "
        "Clustering in Q3 or Q4 points to session degradation - operator fatigue, "
        "equipment wear, or a late interruption - rather than random quality variation."
    )

    report = [
        "# Teleoperation Episode Audit Report",
        "",
        f"Report prefix: `{output_prefix}`",
        f"Dataset: `{loaded.name}`",
        f"Source: `{loaded.source}`",
        f"Minimum episode length threshold: `{summary['min_episode_frames']} frames`",
        "",
        "## Triage Summary",
        "",
        "| Status | Count | Action |",
        "|--------|-------|--------|",
        f"| pass | {pass_count} | Cleared for training queue |",
        f"| review | {review_count} | Human review recommended before training |",
        f"| flag | {flag_count} | Re-collect if possible; exclude from training |",
        "",
        f"Total episodes inspected: `{summary['episode_count']}`  ",
        f"Total frames inspected: `{summary['frame_count']}`  ",
        f"Median episode length: `{summary['median_episode_length']:.1f} frames`",
        "",
        "## Session Pattern",
        "",
        session_note,
        "",
        markdown_table(session_rows) if session_rows else "_Fewer than 4 episodes - session pattern analysis not available._",
        "",
        "## Flagged Episodes",
        "",
        "Episodes requiring action before entering the training queue:",
        "",
        markdown_table(flagged_rows[:20]) if flagged_rows else "_No flagged episodes._",
        "",
        "## Flag Reference",
        "",
        *flag_legend,
        "",
        "## Missing Values",
        "",
        "Fields with missing data across the full dataset:",
        "",
        markdown_table(missing_rows[:20]) if missing_rows else "_No missing values detected._",
        "",
        "## Detected Schema",
        "",
        f"- Episode column: `{summary['episode_col']}`",
        f"- Timestamp column: `{summary['timestamp_col']}`",
        f"- Action column: `{summary['action_col']}`",
        "",
        "## Loader Notes",
        "",
        *[f"- {note}" for note in loaded.notes],
        "",
        "## Threshold Reference",
        "",
        f"- **Minimum episode length**: `{summary['min_episode_frames']} frames` - set via `--min-episode-frames`; calibrate to your task's minimum expected demonstration length",
        f"- **Long episode threshold**: `{summary['long_episode_threshold']} frames` - computed as {LONG_EPISODE_FACTOR:.0f}x median episode length",
        f"- **Timestamp gap**: flags inter-frame gaps larger than {TIMESTAMP_GAP_FACTOR:.0f}x the median gap for that episode",
        f"- **Stuck action run**: flags sequences of {STUCK_ACTION_RUN_LENGTH}+ frames with near-zero action delta (<= {STUCK_ACTION_DELTA})",
    ]

    md_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"\nTriage summary:")
    print(f"  pass:   {pass_count}")
    print(f"  review: {review_count}")
    print(f"  flag:   {flag_count}")
    print(f"\nSample episodes:")
    print(episode_audit.head().to_string(index=False))


if __name__ == "__main__":
    main()
