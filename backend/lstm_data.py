"""
lstm_data.py
============
Build per-node sliding-window sequence datasets for the LSTM burst predictor.

ARCHITECTURE
------------
For every node at every simulation timestep t we:
  1. Collect the last SEQ_LEN pressure snapshots  → shape (SEQ_LEN, N_FEATURES)
  2. Ask: is this node actively leaking between t+1 and t+HORIZON?
     label = 1  →  "burst will be / is active within the next HORIZON steps"
     label = 0  →  "no burst in next HORIZON steps for this node"

The model therefore learns both:
  • Early warning  (label flips 0→1 as the horizon approaches the burst)
  • Confirmed leak (all HORIZON steps are label=1)

FEATURES  (per node per timestep, 10 total — all RELATIVE)
------------------------------------------------------------
  pressure_abs       raw current pressure
  pressure_dev       deviation from network mean at this timestep
  pressure_zscore    standardised deviation
  pressure_pct_rank  fraction of nodes with HIGHER pressure  (1.0 = most suspect)
  net_mean           network average pressure
  net_std            network std
  net_min            network minimum pressure
  net_max            network maximum pressure
  pressure_ratio     pressure / net_mean
  pressure_norm      (pressure - net_min) / (net_max - net_min)

These features generalise to every BIWS junction because they encode the
ANOMALY PATTERN, not node identity.
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# HYPER-PARAMETERS
# ──────────────────────────────────────────────
SEQ_LEN         = 10   # look-back window (timesteps)
HORIZON         = 5    # predict this many steps into the future
NEG_SAMPLE_RATE = 10   # non-leaking nodes sampled per window position

N_FEATURES = 10        # must match implementation of _node_features below


# ──────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────
def _node_features(pressure_row: np.ndarray, node_idx: int) -> np.ndarray:
    """
    10 relative features for ONE node at ONE timestep.
    pressure_row : shape (n_nodes,) — pressures of all junctions at this ts.
    """
    p    = float(pressure_row[node_idx])
    mu   = float(pressure_row.mean())
    sig  = float(pressure_row.std()) + 1e-6
    mn   = float(pressure_row.min())
    mx   = float(pressure_row.max())
    rng  = mx - mn + 1e-6
    rank = float((pressure_row > p).sum()) / len(pressure_row)
    return np.array([
        p,
        p - mu,
        (p - mu) / sig,
        rank,
        mu,
        sig,
        mn,
        mx,
        p / (mu + 1e-6),
        (p - mn) / rng,
    ], dtype=np.float32)


def _all_node_features(pressure_row: np.ndarray) -> np.ndarray:
    """
    Compute 10 relative features for EVERY node at once.
    Returns shape (n_nodes, N_FEATURES) — vectorized version of _node_features.
    """
    n    = len(pressure_row)
    mu   = pressure_row.mean()
    sig  = pressure_row.std() + 1e-6
    mn   = pressure_row.min()
    mx   = pressure_row.max()
    rng  = mx - mn + 1e-6
    rank = np.array([(pressure_row > p).sum() for p in pressure_row],
                    dtype=np.float32) / n
    out = np.column_stack([
        pressure_row,
        pressure_row - mu,
        (pressure_row - mu) / sig,
        rank,
        np.full(n, mu, dtype=np.float32),
        np.full(n, sig, dtype=np.float32),
        np.full(n, mn, dtype=np.float32),
        np.full(n, mx, dtype=np.float32),
        pressure_row / (mu + 1e-6),
        (pressure_row - mn) / rng,
    ])
    return out.astype(np.float32)   # (n_nodes, 10)


# ──────────────────────────────────────────────
# DATASET BUILDER
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


def build_lstm_dataset(
    base_dir:  str,
    scenarios: Optional[List[int]] = None,
    seed:      int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for LSTM training.

    Returns
    -------
    X : float32  (N, SEQ_LEN, N_FEATURES)
    y : float32  (N,)   0.0 or 1.0
        1.0  = this node will be in "leak active" state within the next HORIZON steps
        0.0  = no leak in next HORIZON steps
    """
    rng = np.random.default_rng(seed)

    if scenarios is None:
        scenarios = [sid for sid, _ in find_scenario_dirs(base_dir)]
    if not scenarios:
        raise ValueError(f"No scenarios found under {base_dir}")

    X_list: List[np.ndarray] = []   # each (SEQ_LEN, N_FEATURES)
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
        T            = len(press_df)   # total timesteps in this scenario

        # ── Build feature matrix: (T, n_nodes, N_FEATURES) ──────────────
        feat_all = []   # list of (n_nodes, N_FEATURES) arrays
        for ts in press_df.index:
            p_row = press_df.loc[ts].to_numpy(dtype=np.float32)
            feat_all.append(_all_node_features(p_row))
        feat_all = np.stack(feat_all)   # (T, n_nodes, N_FEATURES)

        # ── Per-node binary label array: (T, n_nodes) ───────────────────
        # label[t, i] = 1 if node i is the leaking node at timestep t
        labels_arr = np.zeros((T, n_nodes), dtype=np.float32)
        ts_list    = list(press_df.index)
        ts_to_t    = {ts: t for t, ts in enumerate(ts_list)}

        for _, row in lb_df.iterrows():
            ts        = str(row["Timestamp"]).strip()
            is_leak   = float(row.get("Label", 0)) == 1.0
            leak_node = str(row.get("Leak_node", "")).strip()
            if is_leak and leak_node and leak_node in node_idx_map and ts in ts_to_t:
                t        = ts_to_t[ts]
                node_idx = node_idx_map[leak_node]
                labels_arr[t, node_idx] = 1.0

        # ── Identify the leaking node for this scenario (if any) ─────────
        leak_node_indices = np.where(labels_arr.sum(axis=0) > 0)[0]

        # ── Sliding window sampling ──────────────────────────────────────
        for t in range(SEQ_LEN, T):
            # Window: timesteps [t-SEQ_LEN, t)  →  predict [t, t+HORIZON)
            future_end = min(t + HORIZON, T)
            window_feat = feat_all[t - SEQ_LEN : t]   # (SEQ_LEN, n_nodes, N_FEATURES)

            # ── Positive samples: leaking nodes ──────────────────────────
            for node_i in leak_node_indices:
                future_label = labels_arr[t:future_end, node_i].max()  # 1 if leaks within horizon
                seq = window_feat[:, node_i, :]   # (SEQ_LEN, N_FEATURES)
                X_list.append(seq)
                y_list.append(float(future_label))

            # ── Negative samples: random non-leaking nodes ────────────────
            non_leak_idxs = [i for i in range(n_nodes) if i not in set(leak_node_indices)]
            if non_leak_idxs:
                neg_idxs = rng.choice(non_leak_idxs,
                                      size=min(NEG_SAMPLE_RATE, len(non_leak_idxs)),
                                      replace=False)
                for node_i in neg_idxs:
                    seq = window_feat[:, node_i, :]
                    X_list.append(seq)
                    y_list.append(0.0)

    if not X_list:
        raise RuntimeError("No sequences built. Check BiWSData has Label and Leak_node columns.")

    X = np.stack(X_list)              # (N, SEQ_LEN, N_FEATURES)
    y = np.array(y_list, dtype=np.float32)
    return X, y


if __name__ == "__main__":
    print("Building LSTM dataset from BiWSData ...")
    X, y = build_lstm_dataset("./BiWSData")
    u, c = np.unique(y, return_counts=True)
    print(f"X shape : {X.shape}  (samples × seq_len × features)")
    print(f"y shape : {y.shape}")
    for cls, cnt in zip(u, c):
        print(f"  label {int(cls)}: {cnt} ({100*cnt/len(y):.1f}%)")
