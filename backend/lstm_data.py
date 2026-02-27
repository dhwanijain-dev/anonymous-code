"""
lstm_data.py
============
Sliding-window regression dataset for the LSTM time-to-burst predictor.

TARGET (regression)
-------------------
For each node at each window position t the target is:

    time_to_burst = steps until this node first enters a "leak active" state

    t < leak_start_step  →  time_to_burst = leak_start_step - t   (future burst)
    t >= leak_start_step →  time_to_burst = 0                       (bursting now)
    no leak in scenario  →  time_to_burst = HORIZON                 (no near-term risk)

At inference, nodes with the LOWEST predicted time-to-burst are the most urgent.
Nodes with time_to_burst ≈ HORIZON have no immediate risk.

WHY THIS GENERALISES TO ALL 2,859 NODES
----------------------------------------
Features are all RELATIVE (no node ID encoded). The LSTM learns:
    "pressure falling fast relative to network → burst imminent (low value)"
    "pressure stable and normal             → no near-term risk (high value ≈ HORIZON)"

FEATURES (10 per node per timestep — fully vectorized)
-------------------------------------------------------
    pressure_abs       raw current pressure
    pressure_dev       deviation from network mean
    pressure_zscore    standardised deviation
    pressure_pct_rank  fraction of nodes with HIGHER pressure  (0→1, 1=lowest=most suspect)
    net_mean           network average pressure at this timestep
    net_std            network std
    net_min / net_max  network pressure extremes
    pressure_ratio     pressure / net_mean
    pressure_norm      (pressure - net_min) / (net_max - net_min)
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# HYPER-PARAMETERS
# ──────────────────────────────────────────────
SEQ_LEN          = 10   # look-back window length (timesteps)
HORIZON          = 20   # max prediction window; "no risk" label value
NEG_SAMPLE_RATE  = 5    # non-leaking node samples per window position
N_FEATURES       = 10   # must match feature extraction below


# ──────────────────────────────────────────────
# VECTORISED FEATURE EXTRACTION
# ──────────────────────────────────────────────
def _all_node_features(pressure_row: np.ndarray) -> np.ndarray:
    """
    Compute 10 relative features for EVERY node at once.
    Input  : pressure_row  shape (N_NODES,)
    Output : feature_mat   shape (N_NODES, N_FEATURES)  float32
    """
    n   = len(pressure_row)
    mu  = pressure_row.mean()
    sig = pressure_row.std() + 1e-6
    mn  = pressure_row.min()
    mx  = pressure_row.max()
    rng = mx - mn + 1e-6

    # Vectorised percentile rank — O(N log N) via argsort
    # rank_pos[i] = fraction of nodes with HIGHER pressure than node i
    order    = np.argsort(pressure_row)[::-1]          # descending
    rank_pos = np.empty(n, dtype=np.float32)
    rank_pos[order] = np.arange(n, dtype=np.float32) / n

    return np.column_stack([
        pressure_row,                           # pressure_abs
        pressure_row - mu,                      # pressure_dev
        (pressure_row - mu) / sig,              # pressure_zscore
        rank_pos,                               # pressure_pct_rank
        np.full(n, mu,  dtype=np.float32),      # net_mean
        np.full(n, sig, dtype=np.float32),      # net_std
        np.full(n, mn,  dtype=np.float32),      # net_min
        np.full(n, mx,  dtype=np.float32),      # net_max
        pressure_row / (mu + 1e-6),             # pressure_ratio
        (pressure_row - mn) / rng,              # pressure_norm
    ]).astype(np.float32)   # (N_NODES, N_FEATURES)


# ──────────────────────────────────────────────
# SCENARIO DISCOVERY
# ──────────────────────────────────────────────
def find_scenario_dirs(base_dir: str) -> List[Tuple[int, str]]:
    paths = glob.glob(os.path.join(base_dir, "Scenario-*"))
    out = []
    for path in paths:
        try:
            sid = int(os.path.basename(path).split("-")[1])
            out.append((sid, os.path.abspath(path)))
        except (IndexError, ValueError):
            continue
    return sorted(out, key=lambda x: x[0])


# ──────────────────────────────────────────────
# DATASET BUILDER
# ──────────────────────────────────────────────
def build_lstm_dataset(
    base_dir:  str,
    scenarios: Optional[List[int]] = None,
    seed:      int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for LSTM regression training.

    Returns
    -------
    X : float32  shape (N_samples, SEQ_LEN, N_FEATURES)
    y : float32  shape (N_samples,)
        Regression target = time_to_burst in timesteps.
        0       → node is currently bursting
        1..HORIZON-1 → burst in N timesteps
        HORIZON → no leak anticipated in the prediction window
    """
    rng = np.random.default_rng(seed)

    if scenarios is None:
        scenarios = [sid for sid, _ in find_scenario_dirs(base_dir)]
    if not scenarios:
        raise ValueError(f"No scenarios found under {base_dir}")

    X_list: List[np.ndarray] = []
    y_list: List[float]      = []

    for sid in scenarios:
        d       = os.path.join(base_dir, f"Scenario-{sid}")
        np_path = os.path.join(d, "Node_pressures.csv")
        lb_path = os.path.join(d, "Labels.csv")
        if not os.path.exists(np_path) or not os.path.exists(lb_path):
            continue

        try:
            press_df = pd.read_csv(np_path, index_col=0)
            lb_df    = pd.read_csv(lb_path)
        except Exception as ex:
            print(f"  [Scenario-{sid}] skip: {ex}")
            continue

        node_cols    = list(press_df.columns)
        n_nodes      = len(node_cols)
        node_idx_map = {n: i for i, n in enumerate(node_cols)}
        T            = len(press_df)
        ts_list      = list(press_df.index)
        ts_to_t      = {ts: t for t, ts in enumerate(ts_list)}

        # ── Pre-compute (T, N_NODES, N_FEATURES) feature cube ────────────
        feat_cube: List[np.ndarray] = []   # T × (N_NODES, N_FEATURES)
        for ts in ts_list:
            p_row = press_df.loc[ts].to_numpy(dtype=np.float32)
            feat_cube.append(_all_node_features(p_row))
        feat_cube_arr = np.stack(feat_cube)   # (T, N_NODES, N_FEATURES)

        # ── Per-node label array: node_label[t, i] = 1 if node i leaking ─
        node_label = np.zeros((T, n_nodes), dtype=np.float32)
        for _, row in lb_df.iterrows():
            ts        = str(row["Timestamp"]).strip()
            is_leak   = float(row.get("Label", 0)) == 1.0
            leak_node = str(row.get("Leak_node", "")).strip()
            if is_leak and leak_node and leak_node in node_idx_map and ts in ts_to_t:
                t_idx = ts_to_t[ts]
                node_label[t_idx, node_idx_map[leak_node]] = 1.0

        # ── Find leak node(s) for this scenario ───────────────────────────
        # leak_first_step[i] = first timestep where node i is leaking, or T
        leak_first_step = np.full(n_nodes, T, dtype=int)
        for i in range(n_nodes):
            active_steps = np.where(node_label[:, i] == 1.0)[0]
            if len(active_steps) > 0:
                leak_first_step[i] = int(active_steps[0])

        leaking_node_idxs = np.where(leak_first_step < T)[0]   # nodes that burst
        non_leaking_idxs  = np.where(leak_first_step == T)[0]  # nodes that never burst

        # ── Sliding window sampling ───────────────────────────────────────
        for t in range(SEQ_LEN, T):
            window = feat_cube_arr[t - SEQ_LEN : t]   # (SEQ_LEN, N_NODES, N_FEATURES)

            # ── Leaking nodes — regression target from time_to_burst ──────
            for node_i in leaking_node_idxs:
                lfs = leak_first_step[node_i]
                # time remaining until burst (clipped to 0 if already bursting)
                ttb = float(max(0, lfs - t))
                seq = window[:, node_i, :]    # (SEQ_LEN, N_FEATURES)
                X_list.append(seq)
                y_list.append(ttb)

            # ── Non-leaking nodes — no risk within window (label = HORIZON) ─
            if len(non_leaking_idxs) > 0:
                sampled = rng.choice(
                    non_leaking_idxs,
                    size=min(NEG_SAMPLE_RATE, len(non_leaking_idxs)),
                    replace=False,
                )
                for node_i in sampled:
                    seq = window[:, node_i, :]
                    X_list.append(seq)
                    y_list.append(float(HORIZON))

    if not X_list:
        raise RuntimeError(
            "No sequences built. "
            "Check BiWSData has Label and Leak_node columns in Labels.csv."
        )

    X = np.stack(X_list)              # (N_samples, SEQ_LEN, N_FEATURES)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ──────────────────────────────────────────────
# INFERENCE HELPER (for main.py / lstm_worker)
# ──────────────────────────────────────────────
def build_inference_tensor(
    pressure_history_list: List[np.ndarray],
    biws_node_order: List[str],
) -> "torch.Tensor":   # type: ignore[name-defined]
    """
    Build the (N_NODES, SEQ_LEN, N_FEATURES) input tensor from a rolling
    buffer of pressure snapshots.

    Parameters
    ----------
    pressure_history_list : list of SEQ_LEN arrays, each shape (N_NODES,)
                            ordered oldest → newest.
    biws_node_order       : list of junction IDs in training column order.

    Returns
    -------
    X_t : torch.Tensor  shape (N_NODES, SEQ_LEN, N_FEATURES)

    Example
    -------
    >>> import torch
    >>> from collections import deque
    >>> buf = deque(maxlen=SEQ_LEN)
    >>> # ... fill buf with numpy arrays from simulation ...
    >>> if len(buf) == SEQ_LEN:
    ...     X_t = build_inference_tensor(list(buf), BIWS_NODE_ORDER)
    ...     with torch.no_grad():
    ...         ttb = lstm_model(X_t).numpy()   # (N_NODES,) — time-to-burst in timesteps
    ...     # sort ascending: smallest value = most urgent
    ...     top_idxs = np.argsort(ttb)[:10]
    ...     for idx in top_idxs:
    ...         print(biws_node_order[idx], "→", ttb[idx], "timesteps")
    """
    import torch
    feat_seq = []
    for p_row in pressure_history_list:
        feat_seq.append(_all_node_features(p_row))   # (N_NODES, N_FEATURES)
    X_np = np.stack(feat_seq, axis=1)               # (N_NODES, SEQ_LEN, N_FEATURES)
    return torch.tensor(X_np, dtype=torch.float32)


if __name__ == "__main__":
    X, y = build_lstm_dataset("./BiWSData")
    print(f"X shape  : {X.shape}   (samples × seq_len × features)")
    print(f"y shape  : {y.shape}")
    print(f"y min    : {y.min():.2f}  (0 = bursting now)")
    print(f"y max    : {y.max():.2f}  (HORIZON = {HORIZON} = no near-term risk)")
    print(f"y mean   : {y.mean():.2f}")
    hist, edges = np.histogram(y, bins=[0,1,5,10,15,20,21])
    for lo, hi, c in zip(edges, edges[1:], hist):
        print(f"  [{lo:2.0f}, {hi:2.0f}) : {c} samples")
