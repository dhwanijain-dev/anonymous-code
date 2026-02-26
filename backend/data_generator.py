
import argparse
import os
import random
import time
import numpy as np
import wntr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # optional: hide repeated warnings about unused curves

parser = argparse.ArgumentParser()
parser.add_argument("--inp", type=str, default="BIWS.inp", help="Path to EPANET .inp file")
parser.add_argument("--out", type=str, default="gnn_rf_dataset.npz", help="Output .npz dataset path")
parser.add_argument("--variations", type=int, default=2, help="Number of leak variations per pipe (>=1)")
parser.add_argument("--leak_areas", type=float, nargs="*", default=[1e-5, 5e-5, 1e-4], help="Leak orifice areas (m^2) to sample from")
parser.add_argument("--snapshot_offset", type=int, default=0, help="Seconds after chosen snapshot (unused normally)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

INP_PATH = args.inp
OUT_PATH = args.out
VARIATIONS = args.variations
LEAK_AREAS = list(args.leak_areas)

if not os.path.exists(INP_PATH):
    raise FileNotFoundError(f"INP file not found: {INP_PATH}")

print("Loading network with WNTR:", INP_PATH)
wn_base = wntr.network.WaterNetworkModel(INP_PATH)

# Run baseline simulation (no leaks)
print("Running baseline simulation...")
sim = wntr.sim.EpanetSimulator(wn_base)
baseline_results = sim.run_sim()

# Choose a snapshot time (prefer 1 hour or 1/4 duration if set)
times = baseline_results.node["pressure"].index.values
if len(times) == 0:
    raise RuntimeError("No simulation times returned. Check INP time settings.")
if wn_base.options.time.duration and wn_base.options.time.duration > 0:
    chosen_snapshot_time = int(min(max(3600, wn_base.options.time.duration // 4), wn_base.options.time.duration - 1))
else:
    chosen_snapshot_time = int(times[len(times) // 2])
snap_idx = baseline_results.node["pressure"].index.get_indexer([chosen_snapshot_time], method="nearest")[0]
snapshot_time = baseline_results.node["pressure"].index[snap_idx]
print("Snapshot time chosen (s):", snapshot_time)

# Baseline pressure vector (node order = wn.node_name_list)
node_names = list(wn_base.node_name_list)
num_nodes = len(node_names)
edge_names = list(wn_base.link_name_list)
num_edges = len(edge_names)
print(f"Network: nodes={num_nodes}, edges={num_edges}")

baseline_pressure = baseline_results.node["pressure"].loc[snapshot_time].values.astype(float)

# static node attrs (optionally used)
node_elevations = np.array([wn_base.get_node(n).elevation if hasattr(wn_base.get_node(n), "elevation") else 0.0 for n in node_names], dtype=float)
node_base_demand = np.array([wn_base.get_node(n).base_demand if hasattr(wn_base.get_node(n), "base_demand") else 0.0 for n in node_names], dtype=float)

# Prepare storage
samples = []          # will store residual vectors (num_nodes,)
labels = []           # leaking pipe index (0..num_edges-1)
meta = []             # metadata per sample

pipe_names = edge_names  # all pipes

total_pipes = len(pipe_names)
print(f"Will generate {VARIATIONS} variation(s) per pipe -> approx {total_pipes * VARIATIONS} samples")

start_time = time.time()
sample_count = 0

for p_idx, pipe_name in enumerate(pipe_names[:500]):
    for v in range(VARIATIONS):
        # reload network fresh to avoid carry-over of previous leak
        wn = wntr.network.WaterNetworkModel(INP_PATH)
        link = wn.get_link(pipe_name)
        # choose leak node: start or end node
        leak_node_name = random.choice([link.start_node_name, link.end_node_name])
        node_obj = wn.get_node(leak_node_name)

        # sample leak area randomly
        leak_area = float(random.choice(LEAK_AREAS))

        # set leak time so it is active at snapshot_time
        leak_start = max(1, int(snapshot_time) - 3600)  # 1 hour before
        leak_end = int(snapshot_time) + 3600             # 1 hour after

        # add leak (pressure-dependent)
        node_obj.add_leak(wn, area=leak_area, start_time=leak_start, end_time=leak_end)

        # run simulation
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        # find nearest time index to snapshot_time in this simulation
        t_idx = results.node["pressure"].index.get_indexer([snapshot_time], method="nearest")[0]
        t_snap = results.node["pressure"].index[t_idx]

        # extract node pressures and compute residual (pressure_now - baseline)
        node_press = results.node["pressure"].loc[t_snap].values.astype(float)
        residual = node_press - baseline_pressure   # shape (num_nodes,)

        # optional additional features: global stats
        stats = np.array([residual.mean(), residual.std(), residual.min(), residual.max(), leak_area], dtype=float)

        # final feature vector: residuals concatenated with a few stats (flatten)
        feat = np.concatenate([residual, stats])  # length = num_nodes + 5

        samples.append(feat)
        labels.append(p_idx)
        meta.append({"leak_pipe": pipe_name, "leak_node": leak_node_name, "leak_area": leak_area, "snapshot_time": int(t_snap)})

        sample_count += 1
        if (sample_count % 50) == 0:
            elapsed = time.time() - start_time
            print(f"Generated {sample_count} samples, last pipe {pipe_name}, elapsed {elapsed:.1f}s")

# Convert to arrays and save
X = np.stack(samples, axis=0).astype(np.float32)  # shape (num_samples, num_nodes+5)
y = np.array(labels, dtype=np.int32)
meta_arr = np.array(meta, dtype=object)

np.savez_compressed(OUT_PATH, X=X, y=y, node_names=np.array(node_names, dtype=object),
                    edge_names=np.array(edge_names, dtype=object), meta=meta_arr)

print("Saved dataset to", OUT_PATH)
print("Samples:", X.shape[0], "Feature dim:", X.shape[1])