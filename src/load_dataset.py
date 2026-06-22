from teleop_data import detect_schema, load_public_or_synthetic, markdown_table, parse_args, reports_dir


def main() -> None:
    args = parse_args()
    loaded = load_public_or_synthetic(args.use_synthetic, args.max_rows)
    frame = loaded.frame
    schema = detect_schema(frame)

    print(f"Dataset: {loaded.name}")
    print(f"Source: {loaded.source}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Episode column: {schema['episode_col']}")
    print(f"Timestamp column: {schema['timestamp_col']}")
    print(f"Action column: {schema['action_col']}")
    print(f"Task column: {schema['task_col']}")
    print(f"Observation-like columns: {schema['observation_cols'][:10]}")
    for note in loaded.notes:
        print(f"Note: {note}")

    sample_rows = frame.head(3).astype(str).to_dict(orient="records")
    report = [
        "# Day 1 Schema Summary",
        "",
        f"Dataset: `{loaded.name}`",
        f"Source: `{loaded.source}`",
        f"Rows inspected: `{len(frame)}`",
        f"Column count: `{len(frame.columns)}`",
        "",
        "## Detected Schema",
        "",
        f"- Episode column: `{schema['episode_col']}`",
        f"- Timestamp column: `{schema['timestamp_col']}`",
        f"- Action column: `{schema['action_col']}`",
        f"- Task column: `{schema['task_col']}`",
        f"- Observation-like columns: `{schema['observation_cols'][:10]}`",
        "",
        "## Loader Notes",
        "",
        *[f"- {note}" for note in loaded.notes],
        "",
        "## Sample Rows",
        "",
        markdown_table(sample_rows),
        "",
        "## Plain-English Interpretation",
        "",
        "This script answers the first ML-readiness question: what data do we actually have?",
        "Before model training, we need to know where episodes, observations, actions, timestamps, and task labels live.",
        "",
        "In real teleoperation workflows, unclear schema or missing sequence fields can make later model training and evaluation hard to debug.",
    ]

    output_path = reports_dir() / "day1_schema_summary.md"
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
