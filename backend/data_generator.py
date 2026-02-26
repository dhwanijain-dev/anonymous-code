import wntr
import pandas as pd
import random

wn = wntr.network.WaterNetworkModel("BIWS.inp")
data_rows = []
scenario_id = 0

# -----------------------------
# Baseline (no leak)
# -----------------------------
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
pressure = results.node["pressure"].copy()
# Add metadata
pressure["scenario_id"] = scenario_id
pressure["label"] = 0                     # all timesteps are no‑leak
pressure["leak_pipe"] = "none"
pressure["leak_node"] = "none"
pressure["leak_area"] = 0.0
data_rows.append(pressure)
scenario_id += 1

# -----------------------------
# Leak scenarios
# -----------------------------
num_leaks = 10
leak_areas = [1e-4, 5e-4, 1e-3]

for i in range(num_leaks):
    pipe_name = random.choice(wn.pipe_name_list)
    pipe = wn.get_link(pipe_name)
    leak_node_name = random.choice([pipe.start_node_name, pipe.end_node_name])
    node = wn.get_node(leak_node_name)

    leak_area = random.choice(leak_areas)
    start_time = 3600
    end_time = 7200

    node.add_leak(wn, area=leak_area, start_time=start_time, end_time=end_time)

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    pressure = results.node["pressure"].copy()

    # Create label column: 1 only during leak window
    pressure["label"] = 0
    pressure.loc[start_time:end_time, "label"] = 1   # assuming index is time in seconds

    pressure["scenario_id"] = scenario_id
    pressure["leak_pipe"] = pipe_name
    pressure["leak_node"] = leak_node_name
    pressure["leak_area"] = leak_area

    data_rows.append(pressure)
    scenario_id += 1

    node.remove_leak(wn)

# Concatenate and save
df = pd.concat(data_rows)
df.to_csv("leak_dataset_timestep.csv")
print("✅ Dataset saved with shape", df.shape)