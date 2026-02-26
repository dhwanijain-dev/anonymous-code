import wntr
import time
import threading
import os
import random
import numpy as np
import csv

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================
# PATHS
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INP_FILE = os.path.join(BASE_DIR, "BIWS.inp")
HTML_FILE = os.path.join(BASE_DIR, "../frontend/dashboard.html")
CSV_FILE = os.path.join(BASE_DIR, "simulation_data.csv")

# ============================
# APP
# ============================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================
# DATASET INITIALIZATION
# ============================

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp", "node_id", "age", "corrosion_idx",
            "length", "baseline_pressure", "press_var",
            "soil_corr_idx", "current_live_pressure", "is_leaking"
        ])

# ============================
# STREAM BUFFER & STATIC PROPS
# ============================

stream_buffer = []

wn_init = wntr.network.WaterNetworkModel(INP_FILE)
node_physical_props = {}

for node in wn_init.junction_name_list:
    node_physical_props[node] = {
        "age": random.uniform(1, 80),
        "corrosion_idx": random.uniform(0.1, 1.0),
        "length": random.uniform(10, 1000),
        "avg_pressure": random.uniform(20, 100),
        "press_var": random.uniform(0.5, 10.0),
        "soil_corr_idx": random.uniform(0.1, 1.0)
    }

# ============================
# SIMULATION WORKER (fixed)
# ============================
def extract_graph_edges(wn, results=None):
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
            graph_edges = extract_graph_edges(wn, results)

            snapshot = {
                "nodes": graph_nodes,
                "edges": graph_edges,
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