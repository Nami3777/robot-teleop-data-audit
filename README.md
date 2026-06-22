# Robot Teleoperation Data Audit Lab

## The Problem

Teleoperation data collection is expensive. Operator time, robot time, and facility access are finite. Bad episodes discovered after training - days of compute later - cost double: the original collection and the repeat.

Common failure modes observed in floor operations:

- **Aborted attempts contaminating clean data**: a mid-episode reset produces a short segment that enters the dataset as a valid demonstration
- **Operator hesitation artifacts**: near-stationary action sequences before difficult manipulations - the policy learns to pause at the hard part
- **Logging gaps**: timestamp discontinuities from sensor handoffs or network interruptions that make the sequence unreliable for supervised learning
- **Session degradation**: episode quality drops across a long collection day due to operator fatigue, equipment wear, or calibration drift - but no individual episode looks obviously wrong

These issues are invisible at collection time and expensive at training time.

## What This Does

Pre-training episode triage. Flags data quality issues before episodes enter the training queue.

For each episode:

- Is it long enough to represent a complete task attempt?
- Does it contain operator hesitation patterns?
- Are there timestamp gaps indicating logging interruptions?
- Does it contain missing values in available sensor or action fields?

For the full session:

- Are flagged episodes clustered in a specific time window?
- Does flag rate increase toward the end - a signal of session degradation?

## How to Run

```bash
pip install -r requirements.txt

# Audit public LeRobot data (downloads on first run)
python src/audit_dataset.py

# Set threshold to your task's minimum expected demonstration length
python src/audit_dataset.py --min-episode-frames 40

# Offline mode with synthetic data
python src/audit_dataset.py --use-synthetic

# Keep public-data and synthetic verification reports separate
python src/audit_dataset.py --use-synthetic --output-prefix synthetic_audit

# Optional PyTorch bridge
pip install -r requirements-optional.txt
python src/torch_dataset.py --use-synthetic
```

Default output: `reports/audit_summary.md` and `reports/audit_summary.csv`.

Use `--output-prefix` when you want to preserve multiple runs, for example `reports/synthetic_audit.md` and `reports/synthetic_audit.csv`.

## Triage Status Levels

| Status | Meaning | Recommended action |
|--------|---------|-------------------|
| `pass` | No quality issues detected | Include in training queue |
| `review` | Minor concern - unusual length or hesitation pattern | Human review before training |
| `flag` | Data integrity issue - short episode or timestamp gap | Re-collect if possible; exclude from training |

## Why Threshold Calibration Matters

The default minimum episode length is 10 frames - a conservative floor. For real tasks, set this to match the minimum expected demonstration:

- Pick-and-place: 40-60 frames
- Manipulation sequences: 80-120 frames
- UAV approach and land: 100-200 frames

Use `--min-episode-frames`. An uncalibrated threshold silently accepts aborted attempts - the audit produces noise instead of signal.

## Cross-Domain Applicability

The checks apply to any sequential operational data where episode quality determines downstream analysis quality:

| Check | Robotics | UAV / Flight Test | Aerospace Test | Autonomous Systems |
|---|---|---|---|---|
| Short episode | Aborted attempt, reset | Aborted test run | Incomplete test sequence | Truncated trajectory |
| Long episode | Operator uncertainty | Extended hold pattern | Off-nominal test duration | Unusual behaviour sequence |
| Timestamp gap | Logging interruption | Sensor dropout, datalink loss | Recording gap, sync failure | Perception dropout |
| Stuck action | Operator hesitation | Actuator hold, command freeze | Control surface saturation | Behaviour deadlock |
| Missing values | Sensor field drop | GPS/IMU loss | Instrument failure | Perception gap |

## Project Layout

```text
robot-teleop-data-audit/
|
|-- src/
|   |-- teleop_data.py      Shared helpers: data loading, schema detection, synthetic fallback
|   |-- audit_dataset.py    Main audit: episode triage -> pass / review / flag + session pattern
|   |-- load_dataset.py     Schema inspection: detect column layout, write Day 1 summary
|   `-- torch_dataset.py    PyTorch bridge: load audited frames into DataLoader (optional)
|
|-- examples/
|   `-- synthetic_audit.md  Sample audit output on synthetic data
|
|-- README.md           Project overview, usage, cross-domain applicability table
|-- requirements.txt    Python dependencies
`-- .gitignore          Excludes .venv, __pycache__, reports/, docs/
```

A sample synthetic audit output is included in `examples/synthetic_audit.md`. Generated reports are written to `reports/` and excluded from version control.

## Pipeline Connection

`src/torch_dataset.py` connects audited episodes to a PyTorch DataLoader. It is optional and requires `requirements-optional.txt`. Filter to `pass`-status rows from the audit CSV before training - exclude episodes flagged for data integrity issues.

## What This Does Not Claim

- Does not train or evaluate a model
- Does not reproduce any company's internal data pipeline
- Does not use internal or confidential data
- Public benchmark data (`lerobot/pusht`) is used as a representative episode format - structure matches production formats, task content does not
- Synthetic fallback data is used only for offline code verification
- Thresholds are starting points; calibrate to your task SOP before production use
