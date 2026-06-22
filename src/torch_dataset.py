from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = object
    Dataset = object

from teleop_data import detect_schema, load_public_or_synthetic, parse_args, values_to_numeric_matrix


class TeleopFrameDataset(Dataset):
    """A simple frame-level bridge from audited robot data into PyTorch."""

    def __init__(self, frame, observation_col: str | None, action_col: str | None):
        if observation_col is None:
            raise ValueError("No observation-like column found for PyTorch demo.")
        if action_col is None:
            raise ValueError("No action column found for PyTorch demo.")

        observations = values_to_numeric_matrix(frame[observation_col])
        actions = values_to_numeric_matrix(frame[action_col])
        if observations.shape[1] == 0:
            raise ValueError(f"Observation column {observation_col} could not be converted to numeric values.")
        if actions.shape[1] == 0:
            raise ValueError(f"Action column {action_col} could not be converted to numeric values.")

        valid_mask = np.isfinite(observations).all(axis=1) & np.isfinite(actions).all(axis=1)
        self.observations = torch.tensor(observations[valid_mask], dtype=torch.float32)
        self.actions = torch.tensor(actions[valid_mask], dtype=torch.float32)
        self.observation_col = observation_col
        self.action_col = action_col

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "observation": self.observations[index],
            "action": self.actions[index],
        }


def main() -> None:
    if torch is None:
        raise SystemExit(
            "PyTorch is required for src/torch_dataset.py. "
            "Install optional dependencies with: pip install -r requirements-optional.txt"
        )

    args = parse_args()
    loaded = load_public_or_synthetic(args.use_synthetic, args.max_rows)
    schema = detect_schema(loaded.frame)
    observation_col = first_numeric_observation_column(loaded.frame, schema["observation_cols"])
    action_col = schema["action_col"]

    dataset = TeleopFrameDataset(loaded.frame, observation_col, action_col)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    print(f"Dataset: {loaded.name}")
    print(f"Source: {loaded.source}")
    print(f"Observation column: {dataset.observation_col}")
    print(f"Action column: {dataset.action_col}")
    print(f"Examples in PyTorch Dataset: {len(dataset)}")
    print(f"Observation batch shape: {tuple(batch['observation'].shape)}")
    print(f"Action batch shape: {tuple(batch['action'].shape)}")
    print("Plain English: the DataLoader groups audited robot frames into batches for model training.")
    for note in loaded.notes:
        print(f"Note: {note}")


def first_numeric_observation_column(frame, columns: list[str]) -> str | None:
    for column in columns:
        matrix = values_to_numeric_matrix(frame[column].head(20))
        if matrix.shape[1] > 0 and np.isfinite(matrix).any():
            return column
    return None


if __name__ == "__main__":
    main()
