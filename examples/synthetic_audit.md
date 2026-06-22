# Teleoperation Episode Audit Report

Dataset: `synthetic/teleop_debug`
Source: `synthetic`
Minimum episode length threshold: `10 frames`

## Triage Summary

| Status | Count | Action |
|--------|-------|--------|
| pass | 1 | Cleared for training queue |
| review | 1 | Human review recommended before training |
| flag | 3 | Re-collect if possible; exclude from training |

Total episodes inspected: `5`  
Total frames inspected: `94`  
Median episode length: `24.0 frames`

## Session Pattern

Flag distribution across the collection session by episode order. Clustering in Q3 or Q4 points to session degradation - operator fatigue, equipment wear, or a late interruption - rather than random quality variation.

| session_quartile | episodes | flagged | flag_rate |
| --- | --- | --- | --- |
| Q1 (earliest) | 2 | 2 | 100% |
| Q2 | 1 | 1 | 100% |
| Q3 | 1 | 1 | 100% |
| Q4 (latest) | 1 | 0 | 0% |

## Flagged Episodes

Episodes requiring action before entering the training queue:

| episode_id | frame_count | timestamp_gap_count | longest_stuck_action_run | status | flags |
| --- | --- | --- | --- | --- | --- |
| 0 | 8 | 0 | 0 | flag | short_episode |
| 1 | 24 | 1 | 0 | flag | timestamp_gap |
| 2 | 30 | 0 | 7 | review | repeated_or_stuck_action |
| 3 | 4 | 0 | 0 | flag | short_episode |

## Flag Reference

- **short_episode**: Possible aborted attempt or incomplete task. Verify against minimum task SOP before including in training.
- **long_episode**: Unusual execution time. Check for operator uncertainty, task interruption, or inconsistent segmentation.
- **repeated_or_stuck_action**: Operator hesitation indicator - near-stationary action sequence detected. These segments add low-information frames that a policy can overfit on.
- **timestamp_gap**: Logging interruption or sensor handoff gap detected. Data integrity in this window is uncertain; prefer re-collection.

## Missing Values

Fields with missing data across the full dataset:

_No missing values detected._

## Detected Schema

- Episode column: `episode_index`
- Timestamp column: `timestamp`
- Action column: `action`

## Loader Notes

- Loaded synthetic fallback data.
- Run without `--use-synthetic` to audit public LeRobot data (downloaded on first run).

## Threshold Reference

- **Minimum episode length**: `10 frames` - set via `--min-episode-frames`; calibrate to your task's minimum expected demonstration length
- **Long episode threshold**: `48 frames` - computed as 2x median episode length
- **Timestamp gap**: flags inter-frame gaps larger than 3x the median gap for that episode
- **Stuck action run**: flags sequences of 5+ frames with near-zero action delta (<= 1e-06)
