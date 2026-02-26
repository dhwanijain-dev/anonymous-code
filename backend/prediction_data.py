import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def find_scenario_dirs(base_dir: str) -> List[Tuple[int, str]]:
    """
    Discover scenario folders of the form 'Scenario-<id>' under base_dir.
    Returns a list of (scenario_id, absolute_path), sorted by id.
    """
    pattern = os.path.join(base_dir, "Scenario-*")
    paths = glob.glob(pattern)

    scenarios: List[Tuple[int, str]] = []
    for path in paths:
        name = os.path.basename(path)
        try:
            # Expect 'Scenario-<number>'
            scenario_str = name.split("-")[1]
            scenario_id = int(scenario_str)
        except (IndexError, ValueError):
            continue
        scenarios.append((scenario_id, os.path.abspath(path)))

    scenarios.sort(key=lambda x: x[0])
    return scenarios


def load_scenario_time_series(
    base_dir: str,
    scenario_id: int,
    horizon_steps: int = 0,
) -> pd.DataFrame:
    """
    Load node pressure time series and leak labels for a single scenario.

    - base_dir: Root directory containing 'Scenario-<id>' folders (e.g. Net1OK).
    - scenario_id: Integer ID such as 480 (for folder 'Scenario-480').
    - horizon_steps: Number of future time steps to predict ahead. If > 0,
      the target label is shifted by -horizon_steps so that each row's
      target corresponds to the leak state horizon_steps into the future.

    Returns a DataFrame indexed by Timestamp with columns:
      [node pressure columns..., 'is_leak', 'target_is_leak']
    """
    scenario_dir = os.path.join(base_dir, f"Scenario-{scenario_id}")
    node_path = os.path.join(scenario_dir, "Node_pressures.csv")
    label_path = os.path.join(scenario_dir, "Labels.csv")

    if not os.path.exists(node_path):
        raise FileNotFoundError(f"Missing Node_pressures.csv for scenario {scenario_id}: {node_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing Labels.csv for scenario {scenario_id}: {label_path}")

    # Node pressures: first column is timestamp (unnamed), remaining are nodes
    node_df = pd.read_csv(node_path, index_col=0, parse_dates=True)
    node_df.index.name = "Timestamp"
    node_df = node_df.sort_index()

    # Per-timestep labels for this scenario
    labels_df = pd.read_csv(label_path, parse_dates=["Timestamp"])
    if "Label" not in labels_df.columns:
        raise ValueError(f"'Label' column not found in {label_path}")

    labels_df = labels_df.sort_values("Timestamp")
    labels_df["is_leak"] = (labels_df["Label"] > 0).astype(int)
    labels_df = labels_df[["Timestamp", "is_leak"]].set_index("Timestamp")

    # Inner join to keep only timestamps present in both pressures and labels
    df = node_df.join(labels_df, how="inner")
    df = df.sort_index()

    if df.empty:
        raise ValueError(f"No overlapping timestamps between pressures and labels for scenario {scenario_id}")

    if horizon_steps != 0:
        # Shift label backwards in time to create a "future leak" target
        df["target_is_leak"] = df["is_leak"].shift(-horizon_steps)
        df = df.dropna(subset=["target_is_leak"])
        df["target_is_leak"] = df["target_is_leak"].astype(int)
    else:
        df["target_is_leak"] = df["is_leak"].astype(int)

    return df


def build_dataset(
    base_dir: str,
    scenarios: Optional[List[int]] = None,
    horizon_steps: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build a supervised learning dataset by stacking time-series data
    from multiple scenarios.

    Returns:
      all_df: full DataFrame (for inspection/debugging)
      X: feature matrix (numpy array)
      y: target leak labels (numpy array of 0/1)
    """
    if scenarios is None:
        discovered = find_scenario_dirs(base_dir)
        scenarios = [sid for sid, _ in discovered]

    if not scenarios:
        raise ValueError(f"No scenarios found under {base_dir}")

    frames: List[pd.DataFrame] = []
    for sid in scenarios:
        scenario_df = load_scenario_time_series(
            base_dir=base_dir,
            scenario_id=sid,
            horizon_steps=horizon_steps,
        )
        scenario_df = scenario_df.copy()
        scenario_df["scenario_id"] = sid
        frames.append(scenario_df)

    if not frames:
        raise RuntimeError("No scenario data could be loaded.")

    all_df = pd.concat(frames, axis=0).sort_index()

    # Define features: all numeric columns except label-related and scenario_id
    drop_cols = {"is_leak", "target_is_leak", "scenario_id"}
    feature_cols = [c for c in all_df.columns if c not in drop_cols]

    X = all_df[feature_cols].to_numpy(dtype=np.float32)
    y = all_df["target_is_leak"].to_numpy(dtype=np.int64)

    return all_df, X, y


if __name__ == "__main__":
    DATA_ROOT = "./Net1OK"   # <-- change if needed
    HORIZON_STEPS = 0        # 0 = detect current leak, >0 = future leak prediction

    print("Building dataset...")
    print(f"Data root      : {DATA_ROOT}")
    print(f"Horizon steps  : {HORIZON_STEPS}")

    df, X, y = build_dataset(
        base_dir=DATA_ROOT,
        horizon_steps=HORIZON_STEPS
    )

    print(f"\nLoaded dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    print("\nClass balance (0 = no leak, 1 = leak):")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")