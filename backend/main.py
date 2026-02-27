import wntr
import time
import threading
import os
import random
import numpy as np
import csv
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import joblib
import torch
import torch.nn as nn

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================
# PATHS
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# BIWS network for live simulation (retrained models are based on this network).
INP_FILE = r"C:\Users\jag7b\urban-water-leakage-detection-system\backend\BIWS.inp"

HTML_FILE = os.path.join(BASE_DIR, "../frontend/dashboard.html")

# Pre-trained models
DETECTION_MODEL_FILE = os.path.join(BASE_DIR, "model", "detection.pkl")
LSTM_MODEL_FILE      = os.path.join(BASE_DIR, "model", "lstm_burst.pt")
LSTM_CONFIG_FILE     = os.path.join(BASE_DIR, "model", "lstm_config.pkl")

# ============================
# MODEL FEATURE SCHEMA
# ============================
# The BIWS junction node IDs in the exact order produced by prediction_data.py
# when building the training dataset from BiWSData/ scenarios.
# Loaded dynamically from the network at startup so it always stays in sync.
# Both detection.pkl and prediction.pkl expect pressures in this order.
_wn_schema = wntr.network.WaterNetworkModel(INP_FILE)
BIWS_NODE_ORDER: List[str] = [str(n).strip() for n in _wn_schema.junction_name_list]
N_MODEL_FEATURES: int = len(BIWS_NODE_ORDER)   # 2859 for BIWS

# ============================
# APP
# ============================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================
# DATASET INITIALIZATION
# ============================


# ============================
# STREAM BUFFER & STATIC PROPS
# ============================

stream_buffer = []

wn_init = wntr.network.WaterNetworkModel(INP_FILE)
node_physical_props = {}
pipe_physical_props = {}

# ============================
# LSTM MODEL DEFINITION
# (must match lstm_model.py exactly)
# ============================
class LeakLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ============================
# LOAD MODELS AT STARTUP
# ============================

# Detection model — binary (is there a leak happening NOW?)
leak_model = None
try:
    leak_model = joblib.load(DETECTION_MODEL_FILE)
    print(f"Loaded detection model from: {DETECTION_MODEL_FILE}")
except Exception as e:
    print(f"WARNING: Failed to load detection model at {DETECTION_MODEL_FILE}: {e}")

# LSTM burst predictor — P(node will burst in next HORIZON timesteps)
lstm_model  = None
lstm_config = {}
try:
    lstm_config = joblib.load(LSTM_CONFIG_FILE)
    lstm_model  = LeakLSTM(
        input_size  = lstm_config["input_size"],
        hidden_size = lstm_config["hidden_size"],
        num_layers  = lstm_config["num_layers"],
        dropout     = lstm_config["dropout"],
    )
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_FILE, map_location="cpu", weights_only=True))
    lstm_model.eval()
    print(f"Loaded LSTM burst model from: {LSTM_MODEL_FILE}")
    print(f"  Horizon = {lstm_config.get('horizon', 5)} timesteps ahead")
except Exception as e:
    print(f"WARNING: Failed to load LSTM model: {e}")
    lstm_model = None

for node in wn_init.junction_name_list:
    node_physical_props[node] = {
        "age": random.uniform(1, 80),
        "corrosion_idx": random.uniform(0.1, 1.0),
        "length": random.uniform(10, 1000),
        "avg_pressure": random.uniform(20, 100),
        "press_var": random.uniform(0.5, 10.0),
        "soil_corr_idx": random.uniform(0.1, 1.0)
    }

for pipe_name in getattr(wn_init, "pipe_name_list", []):
    try:
        link = wn_init.get_link(pipe_name)
        pipe_physical_props[pipe_name] = {
            "age": random.uniform(1, 80),
            "corrosion_idx": random.uniform(0.1, 1.0),
            "soil_corr_idx": random.uniform(0.1, 1.0),
            "length": float(getattr(link, "length", random.uniform(10, 2000)) or random.uniform(10, 2000)),
            "diameter": float(getattr(link, "diameter", random.uniform(0.05, 1.0)) or random.uniform(0.05, 1.0)),
        }
    except Exception:
        continue

# ============================
# ROLLING PRESSURE HISTORY BUFFER
# (used by LSTM burst predictor)
# ============================
LSTM_SEQ_LEN = lstm_config.get("seq_len", 10)   # matches lstm_data.SEQ_LEN
# Each element is a numpy array of shape (N_MODEL_FEATURES,) — ordered by BIWS_NODE_ORDER
pressure_history: deque = deque(maxlen=LSTM_SEQ_LEN)

# ============================
# SIMULATION WORKER
# ============================
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    if not y_true:
        return {"n": 0}
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    n = len(y_true)
    acc = (tp + tn) / n if n else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec)) else None
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def _predict_nodes(
    nodes: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Predict node leak probabilities using the trained XGBoost model.
    Returns (prob_by_node_id, pred_by_node_id).
    """
    # NOTE: This helper is no longer used for per-node features. The detection
    # model (detection.pkl) is trained on full-network snapshots (Net1OK), and
    # in the simulation loop we now apply it at the network level instead.
    return {}, {}


def _build_model_feature_vector(
    graph_nodes: List[Dict[str, Any]],
    active_leak_junctions: Optional[set] = None,
) -> np.ndarray:
    """
    Build the N_MODEL_FEATURES-dimensional feature vector that both
    detection.pkl and prediction.pkl expect, matching the schema produced by
    prediction_data.py when trained on BiWSData/:

      columns 0 .. N_MODEL_FEATURES-1 : live pressures for every BIWS junction
        in the order stored in BIWS_NODE_ORDER (= wn.junction_name_list order).
        Any junction not present in the live snapshot gets pressure 0.0.

    Args:
        graph_nodes: list of node dicts from the live simulation snapshot, each
                     with keys 'id' and 'pressure'.
        active_leak_junctions: unused for BIWS (kept for API compatibility).

    Returns:
        numpy array of shape (1, N_MODEL_FEATURES), dtype float32.
    """
    # Build a fast lookup: node_id -> pressure
    pressure_map: Dict[str, float] = {
        str(n["id"]).strip(): float(n["pressure"]) for n in graph_nodes
    }
    # Map every BIWS junction in training-column order; 0.0 if not in snapshot
    feats = [pressure_map.get(nid, 0.0) for nid in BIWS_NODE_ORDER]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def extract_graph_edges(wn, results=None, active_pipe_leaks: Optional[set] = None):
    """
    Extract pipe/pump/valve connections from WNTR
    """
    edges = []
    G = wn.get_graph()

    for u, v, key in G.edges(keys=True):
        try:
            link = wn.get_link(key)

            edge = {
                "id": key,
                "from": str(u),
                "to": str(v),
                "type": link.link_type,
                "length": getattr(link, "length", None),
                "diameter": getattr(link, "diameter", None),
            }

            # Optional: dynamic flow (if simulation results provided)
            if results is not None:
                try:
                    flow = results.link["flowrate"].iloc[-1][key]
                    edge["flowrate"] = float(flow)
                except Exception:
                    edge["flowrate"] = None

            edge["is_leaking"] = int(active_pipe_leaks is not None and key in active_pipe_leaks)
            edges.append(edge)
        except Exception:
            continue

    return edges

def simulation_worker():
    while True:
        try:
            wn = wntr.network.WaterNetworkModel(INP_FILE)
            wn.options.time.duration = 3600
            wn.options.time.hydraulic_timestep = 60
            wn.options.time.report_timestep = 60

            # 1. Synthetic demand noise
            for node in wn.junction_name_list:
                node_obj = wn.get_node(node)
                if node_obj.demand is not None:
                    try:
                        noise = random.uniform(-0.03, 0.05)
                        node_obj.demand = float(node_obj.demand) * (1 + noise)
                    except Exception:
                        pass

            # Track which nodes get a leak during this cycle
            active_leaks = set()
            # Track which pipes get a leak during this cycle
            active_pipe_leaks = set()

            # 2. Risk-based Leakage Event (may set emitter_coefficient)
            for node in wn.junction_name_list:
                props = node_physical_props.get(node)
                if props is None:
                    continue

                normalized_risk_score = (
                    (props["age"] / 80.0) * 0.2 +
                    (props["corrosion_idx"]) * 0.2 +
                    (props["length"] / 1000.0) * 0.1 +
                    (props["avg_pressure"] / 100.0) * 0.1 +
                    (props["press_var"] / 10.0) * 0.2 +
                    (props["soil_corr_idx"]) * 0.2
                )

                leak_threshold = normalized_risk_score * 0.001

                if np.random.random() < leak_threshold:
                    try:
                        leak_node = wn.get_node(node)

                        # Add a WNTR leak at the junction (pressure-dependent)
                        leak_node.add_leak(
                            wn,
                            area=random.uniform(0.0005, 0.003),   # leak size in m^2
                            discharge_coeff=0.75,
                            start_time=0,                         # seconds
                            end_time=3600
                        )

                        # Keep explicitly tracking the actual leaking nodes
                        active_leaks.add(str(node).strip())

                        print(
                            f"WNTR leak added at node {node} | "
                            f"Area={leak_node.leak_area:.6f} m^2 | "
                            f"Risk Score={normalized_risk_score:.6f}"
                        )
                    except Exception as ex:
                        print(f"Failed to set emitter for node {node}: {ex}")

            # 2b. Risk-based pipe leak events (split pipe + add leak junction)
            # Note: We'll label the affected pipe segments as leaking for evaluation.
            for pipe_name in getattr(wn, "pipe_name_list", []):
                props = pipe_physical_props.get(pipe_name)
                if props is None:
                    continue

                normalized_risk_score = (
                    (props["age"] / 80.0) * 0.25 +
                    (props["corrosion_idx"]) * 0.25 +
                    (min(props["length"], 2000.0) / 2000.0) * 0.25 +
                    (props["soil_corr_idx"]) * 0.25
                )
                pipe_leak_threshold = normalized_risk_score * 0.0005

                if np.random.random() < pipe_leak_threshold:
                    suffix = f"{int(time.time() * 1000)}_{random.randint(0, 10**9)}"
                    new_junc = f"LEAK_{pipe_name}_{suffix}"
                    new_pipe = f"{pipe_name}_SPLIT_{suffix}"
                    try:
                        # Insert a junction mid-pipe and keep original pipe name for one segment.
                        wntr.morph.split_pipe(
                            wn,
                            pipe_name_to_split=pipe_name,
                            new_pipe_name=new_pipe,
                            new_junction_name=new_junc,
                            split_at_point=0.5,
                            return_copy=False,
                        )

                        # Add WNTR leak on the newly created junction
                        leak_junc_obj = wn.get_node(new_junc)
                        leak_junc_obj.add_leak(
                            wn,
                            area=random.uniform(0.0005, 0.003),
                            discharge_coeff=0.75,
                            start_time=0,
                            end_time=3600,
                        )

                        # Ensure we can run the node model for this junction as well
                        if new_junc not in node_physical_props:
                            node_physical_props[new_junc] = {
                                "age": props["age"],
                                "corrosion_idx": props["corrosion_idx"],
                                "length": 1.0,
                                "avg_pressure": random.uniform(20, 100),
                                "press_var": random.uniform(0.5, 10.0),
                                "soil_corr_idx": props["soil_corr_idx"],
                            }

                        active_leaks.add(str(new_junc).strip())
                        active_pipe_leaks.add(str(pipe_name).strip())
                        active_pipe_leaks.add(str(new_pipe).strip())
                    except Exception:
                        # Pipe splitting/leak insertion can fail for some networks; ignore safely.
                        continue

            # Run simulation
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()

            # Get latest pressures
            pressure = results.node['pressure'].iloc[-1]
            current_time = time.time()

            # Stream buffer logic for frontend
            nodes = [str(n).strip() for n in pressure.index.tolist()]
            values = pressure.values.tolist()
            
            # Extract (X, Y) coordinates for each node straight from the WNTR network
            coords = []
            # Stream buffer logic for frontend


            graph_nodes = []

            for node_name, live_pressure in pressure.items():
                node_name = str(node_name).strip()

                try:
                    node_obj = wn.get_node(node_name)
                    x, y = node_obj.coordinates if node_obj.coordinates else (0.0, 0.0)
                except Exception:
                    x, y = 0.0, 0.0

                graph_nodes.append({
                    "id": node_name,
                    "x": float(x),
                    "y": float(y),
                    "pressure": float(live_pressure),
                    "is_leaking": int(node_name in active_leaks)
                })

            # Extract edges (pipes / pumps / valves)
            graph_edges = extract_graph_edges(wn, results, active_pipe_leaks=active_pipe_leaks)

            # ------------------------------------------------------------------
            # Network-level leak DETECTION using detection.pkl
            # Binary classifier trained on BIWS pressure snapshots.
            # Output: det_prob (float 0-1) + det_flag (0/1).
            # ------------------------------------------------------------------
            det_prob = None
            det_flag = None
            if leak_model is not None and len(graph_nodes) > 0:
                try:
                    X_det = _build_model_feature_vector(graph_nodes, active_leaks)
                    try:
                        det_prob = float(leak_model.predict_proba(X_det)[0, 1])
                    except Exception:
                        det_prob = float(leak_model.predict(X_det)[0])
                    det_flag = int(det_prob >= 0.5)
                except Exception as ex:
                    print(f"Detection model inference error: {ex}")
                    det_prob = None
                    det_flag = None

            # Attach the same network-level detection result to every node
            for n in graph_nodes:
                n["pred_leak_prob"] = det_prob
                n["pred_is_leaking"] = det_flag

            # Pipe-level prediction: inherit the max of the two endpoint node probs
            node_prob_map = {str(n["id"]).strip(): n.get("pred_leak_prob") for n in graph_nodes}
            for e in graph_edges:
                u = str(e.get("from", "")).strip()
                v = str(e.get("to", "")).strip()
                pu = node_prob_map.get(u)
                pv = node_prob_map.get(v)
                probs = [p for p in [pu, pv] if isinstance(p, (int, float))]
                p_pipe = max(probs) if probs else None
                e["pred_leak_prob"] = _safe_float(p_pipe)
                e["pred_is_leaking"] = int(p_pipe >= 0.5) if isinstance(p_pipe, (int, float)) else None

            # Metrics vs actual leak flags from this simulation cycle
            node_y_true = [int(n.get("is_leaking", 0)) for n in graph_nodes if n.get("pred_is_leaking") is not None]
            node_y_pred = [int(n.get("pred_is_leaking", 0)) for n in graph_nodes if n.get("pred_is_leaking") is not None]
            pipe_y_true = [int(e.get("is_leaking", 0)) for e in graph_edges if e.get("pred_is_leaking") is not None]
            pipe_y_pred = [int(e.get("pred_is_leaking", 0)) for e in graph_edges if e.get("pred_is_leaking") is not None]

            # ------------------------------------------------------------------
            # LSTM BURST PREDICTOR  — P(node will burst within next HORIZON steps)
            # ------------------------------------------------------------------
            # After each simulation cycle we push the current pressure snapshot
            # into a rolling deque of length SEQ_LEN.  Once the buffer is full
            # we build a (N_NODES, SEQ_LEN, 10) feature tensor for ALL 2,859
            # BIWS junctions simultaneously and run a single batched LSTM forward
            # pass — giving per-node burst probabilities in one shot.
            #
            # Features at each timestep (10 per node, all RELATIVE so the model
            # generalises to every junction including unseen ones):
            #   pressure_abs, pressure_dev, pressure_zscore, pressure_pct_rank,
            #   net_mean, net_std, net_min, net_max, pressure_ratio, pressure_norm
            # ------------------------------------------------------------------
            top_burst_nodes: List[Dict] = []

            # Step 1 — push current snapshot into rolling buffer
            p_map_live = {str(n["id"]).strip(): float(n["pressure"]) for n in graph_nodes}
            p_snap     = np.array([p_map_live.get(nid, 0.0) for nid in BIWS_NODE_ORDER],
                                   dtype=np.float32)  # (N_NODES,)
            pressure_history.append(p_snap)

            # Step 2 — run LSTM only when buffer is full
            if lstm_model is not None and len(pressure_history) == LSTM_SEQ_LEN:
                try:
                    # ── Build (N_NODES, SEQ_LEN, 10) feature tensor ──────────
                    # For each timestep in the window, vectorise the 10 features
                    # across all nodes simultaneously.
                    snap_list = list(pressure_history)  # list of SEQ_LEN (N_NODES,) arrays

                    feat_seq = []  # SEQ_LEN × (N_NODES, 10)
                    for p_row in snap_list:
                        n_n  = len(p_row)   # 2859
                        mu   = p_row.mean()
                        sig  = p_row.std() + 1e-6
                        mn   = p_row.min()
                        mx   = p_row.max()
                        rng  = mx - mn + 1e-6

                        # Vectorized percentile rank via argsort — O(N log N)
                        # rank[i] = fraction of nodes with HIGHER pressure than node i
                        # (closer to 1.0 = lowest pressure = most suspect)
                        order      = np.argsort(p_row)[::-1]   # descending pressure order
                        rank_pos   = np.empty(n_n, dtype=np.float32)
                        rank_pos[order] = np.arange(n_n, dtype=np.float32) / n_n

                        ts_feat = np.column_stack([
                            p_row,                          # pressure_abs
                            p_row - mu,                     # pressure_dev
                            (p_row - mu) / sig,             # pressure_zscore
                            rank_pos,                       # pressure_pct_rank (vectorized)
                            np.full(n_n, mu,  dtype=np.float32),  # net_mean
                            np.full(n_n, sig, dtype=np.float32),  # net_std
                            np.full(n_n, mn,  dtype=np.float32),  # net_min
                            np.full(n_n, mx,  dtype=np.float32),  # net_max
                            p_row / (mu + 1e-6),            # pressure_ratio
                            (p_row - mn) / rng,             # pressure_norm
                        ]).astype(np.float32)               # (N_NODES=2859, 10)
                        feat_seq.append(ts_feat)

                    # Stack → (N_NODES, SEQ_LEN, 10)
                    X_lstm = np.stack(feat_seq, axis=1)                 # (N_NODES, SEQ_LEN, 10)
                    X_t    = torch.tensor(X_lstm, dtype=torch.float32)  # (N_NODES, SEQ_LEN, 10)

                    # ── Single batched forward pass ───────────────────────────
                    with torch.no_grad():
                        burst_probs = lstm_model(X_t).numpy()   # (N_NODES,)

                    # ── Rank ALL 2,859 nodes by burst probability ────────────
                    # Every junction goes through the LSTM — we just surface the
                    # top-K highest burst probabilities in the API response.
                    TOP_K        = 10   # expose top-10 in API
                    horizon      = lstm_config.get("horizon", 5)
                    horizon_min  = horizon   # each timestep = 1 minute (60s step)
                    n_scored     = len(burst_probs)   # should always be 2859

                    top_idxs = np.argsort(burst_probs)[::-1][:TOP_K]
                    for idx in top_idxs:
                        nid = BIWS_NODE_ORDER[idx]
                        props = node_physical_props.get(nid, {})
                        risk_score = (
                            (props.get("age", 40) / 80.0)           * 0.2 +
                            props.get("corrosion_idx", 0.5)          * 0.2 +
                            (props.get("length", 100) / 1000.0)     * 0.1 +
                            (props.get("avg_pressure", 60) / 100.0) * 0.1 +
                            (props.get("press_var", 2) / 10.0)      * 0.2 +
                            props.get("soil_corr_idx", 0.5)          * 0.2
                        )
                        top_burst_nodes.append({
                            "node_id":       nid,
                            "burst_prob":    round(float(burst_probs[idx]), 4),
                            "risk_score":    round(float(risk_score), 4),
                            "horizon_min":   horizon_min,
                        })

                    print(f"[LSTM] Scored {n_scored} nodes | "
                          f"top burst_prob={burst_probs[top_idxs[0]]:.3f} "
                          f"at node {BIWS_NODE_ORDER[top_idxs[0]]}")

                except Exception as ex:
                    print(f"LSTM inference error: {ex}")
                    top_burst_nodes = []

            snapshot = {
                "nodes": graph_nodes,
                "edges": graph_edges,
                "metrics": {
                    "nodes": _binary_metrics(node_y_true, node_y_pred),
                    "pipes": _binary_metrics(pipe_y_true, pipe_y_pred),
                    "model_loaded": bool(leak_model is not None),
                },
                "prediction": {
                    "top_burst_nodes": top_burst_nodes,
                    "n_nodes_scored":  len(burst_probs) if 'burst_probs' in dir() else 0,
                    "buffer_ready":    len(pressure_history) == LSTM_SEQ_LEN,
                    "model_loaded":    bool(lstm_model is not None),
                },
                "timestamp": current_time
            }

            stream_buffer.clear()
            stream_buffer.append(snapshot)

            # sleep between cycles
            time.sleep(5)

        except Exception as e:
            print("Simulation Error:", e)
            # slight pause to avoid busy-loop on persistent failure
            time.sleep(1)

# ============================
# SERVER STARTUP
# ============================

@app.on_event("startup")
def start_simulation():
    threading.Thread(target=simulation_worker, daemon=True).start()

@app.get("/")
def home():
    return FileResponse(HTML_FILE)

@app.get("/stream")
def stream_data():
    return {"data": stream_buffer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)