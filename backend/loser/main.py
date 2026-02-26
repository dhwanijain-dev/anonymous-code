import wntr
import time
import threading
import os
import random

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================
# PATHS
# ============================

BASE_DIR = r"C:\Users\sudee\Desktop\loser"
INP_FILE = os.path.join(BASE_DIR, "BIWS.inp")
HTML_FILE = os.path.join(BASE_DIR, "dashboard.html")

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
# STREAM BUFFER
# ============================

stream_buffer = []

# ============================
# SIMULATION WORKER
# ============================

def simulation_worker():

    while True:

        try:

            # Create network model inside loop
            wn = wntr.network.WaterNetworkModel(INP_FILE)

            wn.options.time.duration = 3600
            wn.options.time.hydraulic_timestep = 60
            wn.options.time.report_timestep = 60

            # Synthetic disturbance injection
            for node in wn.junction_name_list:

                node_obj = wn.get_node(node)

                if node_obj.demand is not None:

                    try:
                        noise = random.uniform(-0.03, 0.05)

                        node_obj.demand = float(node_obj.demand) * (1 + noise)

                    except:
                        pass

            # Random synthetic leakage event
            leak_nodes = random.sample(
                wn.junction_name_list,
                k=max(1, int(len(wn.junction_name_list) * 0.02))
            )

            for node in leak_nodes:
                try:
                    wn.get_node(node).emitter_coefficient = random.uniform(20, 60)
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