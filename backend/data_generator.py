
import os
import time
import argparse
import random
import copy
import csv
from typing import List

import numpy as np
import wntr

# -----------------------
# CLI args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", type=str, default="BIWS.inp", help="Path to INP file")
    p.add_argument("--out", type=str, default="simulation_data.csv", help="Output CSV")
    p.add_argument("--cycles", type=int, default=1000, help="Number of simulation cycles to run")
    p.add_argument("--mode", choices=["fast", "realistic"], default="fast",
                   help="fast = EPANET emitter (fastest). realistic = WNTR node.add_leak (pressure-dependent)")
    p.add_argument("--batch", type=int, default=50, help="Number of cycles per CSV flush")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--leak_scale", type=float, default=0.02,
                   help="Multiplier on normalized risk score to make leaks more/less frequent. Increase for more leaks.")
    p.add_argument("--sim_duration", type=int, default=600, help="Simulation duration (seconds) per cycle")
    return p.parse_args()

# -----------------------
# Helper: init node physical props
# -----------------------
def init_node_physical_props(junctions: List[str], rng: np.random.Generator):
    props = {}
    # Use deterministic-ish distributions based on rng
    ages = rng.uniform(1, 80, size=len(junctions))
    corrs = rng.uniform(0.1, 1.0, size=len(junctions))
    lengths = rng.uniform(10, 1000, size=len(junctions))
    avg_press = rng.uniform(20, 100, size=len(junctions))
    press_var = rng.uniform(0.5, 10.0, size=len(junctions))
    soil_corr = rng.uniform(0.1, 1.0, size=len(junctions))

    for i, node in enumerate(junctions):
        props[node] = {
            "age": float(ages[i]),
            "corrosion_idx": float(corrs[i]),
            "length": float(lengths[i]),
            "avg_pressure": float(avg_press[i]),
            "press_var": float(press_var[i]),
            "soil_corr_idx": float(soil_corr[i])
        }
    return props

# -----------------------
# Core generator
# -----------------------
def generate(
    inp_file: str,
    out_csv: str,
    cycles: int = 1000,
    mode: str = "fast",
    batch_size: int = 50,
    seed: int = None,
    leak_scale: float = 0.02,
    sim_duration: int = 600,
):
    # RNGs
    if seed is not None:
        random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Load base network once
    wn_base = wntr.network.WaterNetworkModel(inp_file)

    junctions = wn_base.junction_name_list
    n_nodes = len(junctions)
    if n_nodes == 0:
        raise RuntimeError("No junctions found in the INP file.")

    # Initialize stable physical props (same as your original approach)
    node_physical_props = init_node_physical_props(junctions, rng)

    # Prepare CSV (header if not exists)
    header = [
        "timestamp", "node_id", "age", "corrosion_idx",
        "length", "baseline_pressure", "press_var",
        "soil_corr_idx", "current_live_pressure", "is_leaking"
    ]
    write_header = not os.path.exists(out_csv)
    csv_file = open(out_csv, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(header)
        csv_file.flush()

    rows_buffer = []
    t0 = time.time()

    # Precompute arrays for vectorized risk score calculation
    age_arr = np.array([node_physical_props[n]["age"] for n in junctions], dtype=float)
    corr_arr = np.array([node_physical_props[n]["corrosion_idx"] for n in junctions], dtype=float)
    length_arr = np.array([node_physical_props[n]["length"] for n in junctions], dtype=float)
    avgp_arr = np.array([node_physical_props[n]["avg_pressure"] for n in junctions], dtype=float)
    pv_arr = np.array([node_physical_props[n]["press_var"] for n in junctions], dtype=float)
    soil_arr = np.array([node_physical_props[n]["soil_corr_idx"] for n in junctions], dtype=float)

    for cycle in range(1, cycles + 1):
        cycle_start = time.time()
        # Use a fresh model copy for each cycle (deepcopy is faster than reloading .inp repeatedly)
        wn = copy.deepcopy(wn_base)

        # Tweak simulation times (shorter durations = faster generation)
        wn.options.time.duration = sim_duration
        wn.options.time.hydraulic_timestep = 60
        wn.options.time.report_timestep = 60

        # 1) demand noise (vectorized-ish)
        # Loop needed because node.demand may be None or not numeric
        for n in wn.junction_name_list:
            node_obj = wn.get_node(n)
            if node_obj.demand is not None:
                try:
                    # small multiplicative noise
                    noise = rng.uniform(-0.03, 0.05)
                    node_obj.demand = float(node_obj.demand) * (1.0 + float(noise))
                except Exception:
                    pass

        # 2) risk-based leak decision (vectorized)
        normalized_risk_score = (
            (age_arr / 80.0) * 0.2 +
            (corr_arr) * 0.2 +
            (length_arr / 1000.0) * 0.1 +
            (avgp_arr / 100.0) * 0.1 +
            (pv_arr / 10.0) * 0.2 +
            (soil_arr) * 0.2
        )  # shape (n_nodes,)

        # leak_thresholds, scaled by leak_scale CLI arg
        leak_thresholds = normalized_risk_score * leak_scale  # tweak leak_scale to control freq

        random_draws = rng.random(n_nodes)
        leak_flags = random_draws < leak_thresholds  # boolean array: True => create leak

        # For bookkeeping: sets for CSV detection
        leak_nodes_set = set()

        # Apply leaks (fast vs realistic)
        if mode == "fast":
            # EPANET emitters approach (fast)
            for idx, node_name in enumerate(junctions):
                if leak_flags[idx]:
                    try:
                        wn.get_node(node_name).emitter_coefficient = float(rng.uniform(20, 60))
                        leak_nodes_set.add(node_name)
                    except Exception:
                        pass
        else:
            # realistic mode: use WNTR node.add_leak (pressure-dependent). Slower but physical.
            for idx, node_name in enumerate(junctions):
                if leak_flags[idx]:
                    try:
                        ln = wn.get_node(node_name)
                        ln.add_leak(
                            wn,
                            area=float(rng.uniform(0.0005, 0.003)),
                            discharge_coeff=0.75,
                            start_time=0,
                            end_time=sim_duration
                        )
                        # WNTR stores leak_area (or leak params). Use name in bookkeeping
                        leak_nodes_set.add(node_name)
                    except Exception:
                        pass

        # 3) simulate
        try:
            if mode == "fast":
                sim = wntr.sim.EpanetSimulator(wn)
            else:
                sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()
        except Exception as ex:
            # If simulation fails, skip this cycle but log a short message
            print(f"[Cycle {cycle}] Simulation failed: {ex}")
            # continue to next cycle
            continue

        # get latest pressures (last reported time)
        try:
            pressure = results.node["pressure"].iloc[-1]
        except Exception:
            # if pressure extraction fails skip cycle
            print(f"[Cycle {cycle}] Failed to get pressure results, skipping.")
            continue

        current_time = time.time()

        # batch rows for CSV
        for node_name, live_pressure in pressure.items():
            node_key = str(node_name).strip()
            if node_key in node_physical_props:
                props = node_physical_props[node_key]
                is_leaking = 1 if node_key in leak_nodes_set else 0
                rows_buffer.append([
                    current_time,
                    node_key,
                    props["age"],
                    props["corrosion_idx"],
                    props["length"],
                    props["avg_pressure"],
                    props["press_var"],
                    props["soil_corr_idx"],
                    float(live_pressure),
                    is_leaking
                ])

        # flush buffer in batches to keep memory low and speed high
        if cycle % batch_size == 0 or cycle == cycles:
            csv_writer.writerows(rows_buffer)
            csv_file.flush()
            rows_buffer.clear()

        # Optional lightweight progress print (every 100 cycles)
        if cycle % max(1, cycles // 20) == 0:
            elapsed = time.time() - t0
            avg_per_cycle = elapsed / cycle
            print(f"[{cycle}/{cycles}] elapsed={elapsed:.1f}s avg_cycle={avg_per_cycle:.3f}s total_rows={cycle * n_nodes}")

    csv_file.close()
    total_time = time.time() - t0
    print(f"Done. Generated ~{cycles * n_nodes} rows in {total_time:.1f}s ({(cycles * n_nodes) / total_time:.1f} rows/s)")

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Mode: {args.mode} | Cycles: {args.cycles} | Batch: {args.batch} | Leak scale: {args.leak_scale}")

    generate(
        inp_file=args.inp,
        out_csv=args.out,
        cycles=args.cycles,
        mode=args.mode,
        batch_size=args.batch,
        seed=args.seed,
        leak_scale=args.leak_scale,
        sim_duration=args.sim_duration,
    )