import wntr
import time
import threading
import os
import random
import numpy as np

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================
# PATHS
# ============================

INP_FILE = "BIWS.inp"
HTML_FILE = "../frontend/dashboard.html"

# ============================
# APP
# ============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# STREAM BUFFER & STATIC PROPS
# ============================

stream_buffer = []

# Generate static physical properties per node once
# so they remain consistent across simulation loops
wn_init = wntr.network.WaterNetworkModel(INP_FILE)
node_physical_props = {}

for node in wn_init.junction_name_list:
    node_physical_props[node] = {
        "age": random.uniform(1, 80),                  # Years (0 to 80)
        "corrosion_idx": random.uniform(0.1, 1.0),     # 0.1 (low) to 1.0 (high)
        "length": random.uniform(10, 1000),            # Segment length in meters
        "avg_pressure": random.uniform(20, 100),       # Baseline internal pressure
        "press_var": random.uniform(0.5, 10.0),        # Std dev of pressure
        "soil_corr_idx": random.uniform(0.1, 1.0)      # Soil corrosivity (0.1 to 1.0)
    }


# ============================
# SIMULATION WORKER
# ============================

def simulation_worker():

    while True:
        try:
            wn = wntr.network.WaterNetworkModel(INP_FILE)

            wn.options.time.duration = 3600
            wn.options.time.hydraulic_timestep = 60
            wn.options.time.report_timestep = 60

            # 1. Synthetic disturbance injection (Demand Noise)
            for node in wn.junction_name_list:
                node_obj = wn.get_node(node)
                if node_obj.demand is not None:
                    try:
                        noise = random.uniform(-0.03, 0.05)
                        node_obj.demand = float(node_obj.demand) * (1 + noise)
                    except:
                        pass

            # 2. Risk-based Leakage Event
            for node in wn.junction_name_list:
                props = node_physical_props[node]
                
                # Normalize features roughly to a 0-1 scale and apply arbitrary weights
                # Sum of weights = 1.0 (0.2 + 0.2 + 0.1 + 0.1 + 0.2 + 0.2)
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
                        # Add a leak to this specific vulnerable node
                        wn.get_node(node).emitter_coefficient = random.uniform(20, 60)
                        print(f"Leak triggered at node {node}! Risk Score: {normalized_risk_score:.2f}")
                    except:
                        pass

            # Run simulation
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()

            pressure = results.node['pressure'].iloc[-1]

            nodes = pressure.index.tolist()
            values = pressure.values.tolist()

            risk_nodes = [
                node for node, val in pressure.items()
                if val < random.uniform(0.8, 1.5)
            ]

            snapshot = {
                "nodes": nodes,
                "pressure": values,
                "risk_nodes": risk_nodes,
                "timestamp": time.time()
            }

            stream_buffer.clear()
            stream_buffer.append(snapshot)

            time.sleep(5)

        except Exception as e:
            print("Simulation Error:", e)

# ============================
# BACKGROUND THREAD START
# ============================

@app.on_event("startup")
def start_simulation():
    threading.Thread(target=simulation_worker, daemon=True).start()

# ============================
# ROUTES
# ============================

@app.get("/")
def home():
    return FileResponse(HTML_FILE)

@app.get("/stream")
def stream_data():
    return {"data": stream_buffer}

# ============================
# RUN SERVER
# ============================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)