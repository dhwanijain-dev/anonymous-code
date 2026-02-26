import wntr
import time
import threading
import os
import random
import numpy as np
import csv
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================
# PATHS
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Override and always use the original BIWS.inp network for live simulation.
INP_FILE = os.path.join(BASE_DIR, "BIWS.inp")

HTML_FILE = os.path.join(BASE_DIR, "../frontend/dashboard.html")

# Pre-trained models (provided separately)
DETECTION_MODEL_FILE = os.path.join(BASE_DIR, "model", "detection.pkl")
PREDICTION_MODEL_FILE = os.path.join(BASE_DIR, "model", "prediction.pkl")

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

# Detection model (current leak classification)
leak_model = None
try:
    leak_model = joblib.load(DETECTION_MODEL_FILE)
    print(f"Loaded detection model from: {DETECTION_MODEL_FILE}")
except Exception as e:
    print(f"WARNING: Failed to load detection model at {DETECTION_MODEL_FILE}: {e}")
    leak_model = None

# Prediction model (future leak forecasting at network level)
prediction_model = None
try:
    prediction_model = joblib.load(PREDICTION_MODEL_FILE)
    print(f"Loaded prediction model from: {PREDICTION_MODEL_FILE}")
except Exception as e:
    print(f"WARNING: Failed to load prediction model at {PREDICTION_MODEL_FILE}: {e}")
    prediction_model = None

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
# SIMULATION WORKER (fixed)
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
    if leak_model is None:
        return {}, {}

    rows = []
    ids = []
    for n in nodes:
        node_id = str(n.get("id", "")).strip()
        if not node_id:
            continue
        props = node_physical_props.get(node_id)
        if props is None:
            # For dynamically-created nodes (e.g., pipe-split leak junctions), we skip if no props
            continue
        ids.append(node_id)
        rows.append(
            {
                "age": props.get("age", 0.0),
                "corrosion_idx": props.get("corrosion_idx", 0.0),
                "length": props.get("length", 0.0),
                # model.py trained on 'baseline_pressure' column; our props store avg_pressure
                "baseline_pressure": props.get("avg_pressure", 0.0),
                "press_var": props.get("press_var", 0.0),
                "soil_corr_idx": props.get("soil_corr_idx", 0.0),
                "current_live_pressure": n.get("pressure", 0.0),
            }
        )

    if not rows:
        return {}, {}

    X = pd.DataFrame(rows).fillna(0.0)
    try:
        proba = leak_model.predict_proba(X)[:, 1]
    except Exception:
        # Some models may not support predict_proba; fall back to predict
        preds = leak_model.predict(X)
        prob_by_id = {i: float(p) for i, p in zip(ids, preds)}
        pred_by_id = {i: int(p >= threshold) for i, p in zip(ids, preds)}
        return prob_by_id, pred_by_id

    prob_by_id = {i: float(p) for i, p in zip(ids, proba)}
    pred_by_id = {i: int(p >= threshold) for i, p in zip(ids, proba)}
    return prob_by_id, pred_by_id


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

            # Model predictions (nodes)
            prob_by_node, pred_by_node = _predict_nodes(graph_nodes, threshold=0.5)
            for n in graph_nodes:
                nid = str(n.get("id", "")).strip()
                if nid in prob_by_node:
                    n["pred_leak_prob"] = prob_by_node[nid]
                    n["pred_is_leaking"] = pred_by_node[nid]
                else:
                    n["pred_leak_prob"] = None
                    n["pred_is_leaking"] = None

            # Model predictions (pipes): approximate from endpoint node probabilities
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

            # Optional: use prediction_model for network-level future leak risk
            future_prob = None
            future_flag = None
            if prediction_model is not None and graph_nodes:
                try:
                    pressures = np.array([n["pressure"] for n in graph_nodes], dtype=float)
                    node_preds = np.array(
                        [int(n.get("pred_is_leaking", 0)) for n in graph_nodes],
                        dtype=int,
                    )
                    features = {
                        "pressure_mean": float(pressures.mean()),
                        "pressure_std": float(pressures.std()),
                        "pressure_min": float(pressures.min()),
                        "pressure_max": float(pressures.max()),
                        "num_nodes": float(len(graph_nodes)),
                        "frac_nodes_pred_leak": float(node_preds.sum() / len(graph_nodes)),
                    }
                    X_future = pd.DataFrame([features])
                    try:
                        future_prob = float(prediction_model.predict_proba(X_future)[0, 1])
                    except Exception:
                        future_prob = float(prediction_model.predict(X_future)[0])
                    future_flag = int(future_prob >= 0.5)
                except Exception as _:
                    future_prob = None
                    future_flag = None

            snapshot = {
                "nodes": graph_nodes,
                "edges": graph_edges,
                "metrics": {
                    "nodes": _binary_metrics(node_y_true, node_y_pred),
                    "pipes": _binary_metrics(pipe_y_true, pipe_y_pred),
                    "model_loaded": bool(leak_model is not None),
                },
                "prediction": {
                    "future_leak_prob": future_prob,
                    "future_leak_flag": future_flag,
                    "model_loaded": bool(prediction_model is not None),
                },
                "timestamp": current_time
            }
            print(graph_nodes[0])
            print(graph_edges[0])
            print(current_time)
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