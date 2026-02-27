"""
biws_data_generator.py
======================
Generate Net1OK-style scenario folders for the BIWS network so that:
  - prediction_data.py can load them unchanged
  - model.py  (detection.pkl)  can be retrained
  - prediction_model.py (prediction.pkl) can be retrained

Each scenario folder  BiWSData/Scenario-<N>/  contains:
  Node_pressures.csv   — rows=timestamps (ISO), cols=junction node IDs
  Labels.csv           — columns: Timestamp, Label  (0 = no leak, 1 = leak)

Usage
-----
  py biws_data_generator.py                       # 100 scenarios, fast mode
  py biws_data_generator.py --scenarios 200       # more scenarios
  py biws_data_generator.py --mode realistic      # pressure-dependent leaks
  py biws_data_generator.py --seed 42             # reproducible
  py biws_data_generator.py --scenarios 50 --duration 3600   # 1 h per scenario
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import wntr

# ──────────────────────────────────────────────
# CONFIG DEFAULTS
# ──────────────────────────────────────────────
DEFAULT_INP      = "BIWS.inp"
DEFAULT_OUT_DIR  = "BiWSData"
DEFAULT_SCENARIOS = 100
DEFAULT_DURATION  = 3600         # seconds per scenario  (1 h → 60 time-steps @ 60 s)
DEFAULT_TIMESTEP  = 60           # hydraulic + report timestep (seconds)
DEFAULT_MODE      = "fast"       # "fast" = EPANET emitter, "realistic" = WNTR add_leak
DEFAULT_LEAK_SCALE = 0.015       # controls leak frequency; raise for more leaks


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate BIWS training scenarios for leak detection/prediction models."
    )
    p.add_argument("--inp",       type=str,   default=DEFAULT_INP,
                   help=f"Path to BIWS INP file (default: {DEFAULT_INP})")
    p.add_argument("--out",       type=str,   default=DEFAULT_OUT_DIR,
                   help=f"Output root directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--scenarios", type=int,   default=DEFAULT_SCENARIOS,
                   help=f"Number of scenarios to generate (default: {DEFAULT_SCENARIOS})")
    p.add_argument("--duration",  type=int,   default=DEFAULT_DURATION,
                   help=f"Simulation duration in seconds per scenario (default: {DEFAULT_DURATION})")
    p.add_argument("--timestep",  type=int,   default=DEFAULT_TIMESTEP,
                   help=f"Hydraulic + report timestep in seconds (default: {DEFAULT_TIMESTEP})")
    p.add_argument("--mode",      choices=["fast", "realistic"], default=DEFAULT_MODE,
                   help="fast = EPANET emitter (fastest). realistic = WNTR add_leak (slower, physical).")
    p.add_argument("--leak_scale", type=float, default=DEFAULT_LEAK_SCALE,
                   help=f"Leak frequency multiplier (default: {DEFAULT_LEAK_SCALE}). Raise for more leaks.")
    p.add_argument("--seed",      type=int,   default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--start",     type=int,   default=1,
                   help="Starting scenario number (useful for resuming; default: 1).")
    return p.parse_args()


def init_phys_props(junctions: list, rng: np.random.Generator) -> dict:
    """Assign stable physical properties to every junction once per run."""
    n = len(junctions)
    ages      = rng.uniform(1, 80,   size=n)
    corrs     = rng.uniform(0.1, 1.0, size=n)
    lengths   = rng.uniform(10, 1000, size=n)
    avg_press = rng.uniform(20, 100,  size=n)
    press_var = rng.uniform(0.5, 10.0, size=n)
    soil_corr = rng.uniform(0.1, 1.0, size=n)
    props = {}
    for i, node in enumerate(junctions):
        props[node] = {
            "age":           float(ages[i]),
            "corrosion_idx": float(corrs[i]),
            "length":        float(lengths[i]),
            "avg_pressure":  float(avg_press[i]),
            "press_var":     float(press_var[i]),
            "soil_corr_idx": float(soil_corr[i]),
        }
    return props


def risk_scores(junctions: list, props: dict) -> np.ndarray:
    """Vectorised normalised risk score in [0, 1] for every junction."""
    age_arr  = np.array([props[n]["age"]           for n in junctions])
    corr_arr = np.array([props[n]["corrosion_idx"]  for n in junctions])
    len_arr  = np.array([props[n]["length"]         for n in junctions])
    avgp_arr = np.array([props[n]["avg_pressure"]   for n in junctions])
    pv_arr   = np.array([props[n]["press_var"]      for n in junctions])
    soil_arr = np.array([props[n]["soil_corr_idx"]  for n in junctions])
    return (
        (age_arr  / 80.0)   * 0.2 +
        corr_arr            * 0.2 +
        (len_arr  / 1000.0) * 0.1 +
        (avgp_arr / 100.0)  * 0.1 +
        (pv_arr   / 10.0)   * 0.2 +
        soil_arr            * 0.2
    )


def run_scenario(
    wn_base:    wntr.network.WaterNetworkModel,
    junctions:  list,
    phys_props: dict,
    risk_arr:   np.ndarray,
    rng:        np.random.Generator,
    scenario_id: int,
    out_dir:    str,
    duration:   int,
    timestep:   int,
    mode:       str,
    leak_scale: float,
) -> bool:
    """
    Simulate one scenario, save Node_pressures.csv + Labels.csv.
    Returns True on success, False if simulation failed.
    """
    wn = copy.deepcopy(wn_base)
    wn.options.time.duration          = duration
    wn.options.time.hydraulic_timestep = timestep
    wn.options.time.report_timestep   = timestep

    # 1. Demand noise
    for n in wn.junction_name_list:
        node_obj = wn.get_node(n)
        if node_obj.demand is not None:
            try:
                node_obj.demand = float(node_obj.demand) * (1.0 + float(rng.uniform(-0.03, 0.05)))
            except Exception:
                pass

    # 2. Decide whether this scenario has leaks at all.
    #    ~50% of scenarios are "no-leak" (all rows label=0) and ~50% inject leaks.
    #    For leak scenarios, leaks start at a random mid-point so the
    #    pre-leak rows are label=0 and post-start rows are label=1 —
    #    matching the per-timestep label variation seen in Net1OK data.
    has_leak_scenario = rng.random() < 0.5
    n_timesteps = duration // timestep          # e.g. 60 for 3600s / 60s

    # Leak start index: random point in [10%, 50%] of the scenario
    if has_leak_scenario:
        leak_start_step = int(rng.integers(
            max(1, n_timesteps // 10),
            max(2, n_timesteps // 2),
        ))
        leak_start_time = leak_start_step * timestep  # seconds into the simulation
    else:
        leak_start_step = n_timesteps   # leaks never start
        leak_start_time = duration + 1

    leak_set: set = set()

    if has_leak_scenario:
        # Risk-based decision on which nodes leak
        thresholds = risk_arr * leak_scale
        draws      = rng.random(len(junctions))
        leak_flags = draws < thresholds

        if mode == "fast":
            for idx, node_name in enumerate(junctions):
                if leak_flags[idx]:
                    try:
                        wn.get_node(node_name).emitter_coefficient = float(rng.uniform(20, 60))
                        leak_set.add(node_name)
                    except Exception:
                        pass
        else:  # realistic
            for idx, node_name in enumerate(junctions):
                if leak_flags[idx]:
                    try:
                        ln = wn.get_node(node_name)
                        ln.add_leak(
                            wn,
                            area=float(rng.uniform(0.0005, 0.003)),
                            discharge_coeff=0.75,
                            start_time=leak_start_time,
                            end_time=duration,
                        )
                        leak_set.add(node_name)
                    except Exception:
                        pass

        # For fast mode: emitter coefficients are applied for the whole run,
        # but we use the label schema to mark only post-leak-start rows as 1.
        # If no node actually got a leak (all tries failed), treat as no-leak.
        if not leak_set:
            has_leak_scenario = False
            leak_start_step = n_timesteps

    # 3. Simulate
    try:
        sim = (wntr.sim.EpanetSimulator(wn) if mode == "fast"
               else wntr.sim.WNTRSimulator(wn))
        results = sim.run_sim()
        pressure_df = results.node["pressure"]   # DataFrame: index=time(s), cols=node_ids
    except Exception as ex:
        print(f"  [Scenario-{scenario_id}] Simulation failed: {ex}")
        return False

    # 4. Build Node_pressures.csv
    #    Rows = timestamps (ISO strings), Columns = junction node IDs only
    junction_set_s = set(junctions)
    pressure_junctions = pressure_df[
        [c for c in pressure_df.columns if str(c).strip() in junction_set_s]
    ]

    # Convert the time-in-seconds index into human-readable ISO timestamps
    base_ts = pd.Timestamp("2024-01-01 00:00:00")
    time_index_seconds = list(pressure_df.index)   # in seconds
    pressure_junctions.index = [
        (base_ts + pd.Timedelta(seconds=int(t))).isoformat(sep=" ")
        for t in time_index_seconds
    ]
    pressure_junctions.index.name = "Timestamp"

    # 5. Build Labels.csv — per-timestep labels
    #    Rows before leak_start_time  → Label 0
    #    Rows at or after that time   → Label 1 (if any leak was injected)
    row_labels = []
    for t_sec in time_index_seconds:
        if has_leak_scenario and int(t_sec) >= leak_start_time:
            row_labels.append(1.0)
        else:
            row_labels.append(0.0)

    labels = pd.DataFrame({
        "Timestamp": pressure_junctions.index.tolist(),
        "Label":     row_labels,
    })

    # 6. Save
    scen_dir = os.path.join(out_dir, f"Scenario-{scenario_id}")
    os.makedirs(scen_dir, exist_ok=True)
    pressure_junctions.to_csv(os.path.join(scen_dir, "Node_pressures.csv"))
    labels.to_csv(os.path.join(scen_dir, "Labels.csv"), index=False)

    return True


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        print(f"Random seed: {args.seed}")
    else:
        rng = np.random.default_rng()

    print(f"Loading network: {args.inp}")
    wn_base  = wntr.network.WaterNetworkModel(args.inp)
    junctions = wn_base.junction_name_list
    print(f"  Junctions : {len(junctions)}")
    print(f"  Pipes     : {len(wn_base.pipe_name_list)}")

    print(f"\nInitialising physical properties...")
    phys_props = init_phys_props(junctions, rng)
    risk_arr   = risk_scores(junctions, phys_props)

    os.makedirs(args.out, exist_ok=True)

    total     = args.scenarios
    start_id  = args.start
    success   = 0
    t0        = time.time()

    print(f"\nGenerating {total} scenarios → {args.out}/")
    print(f"  Mode: {args.mode} | Duration: {args.duration}s | Timestep: {args.timestep}s | Leak scale: {args.leak_scale}")
    print(f"  Starting from Scenario-{start_id}")
    print()

    for i, scenario_id in enumerate(range(start_id, start_id + total), start=1):
        ok = run_scenario(
            wn_base=wn_base,
            junctions=junctions,
            phys_props=phys_props,
            risk_arr=risk_arr,
            rng=rng,
            scenario_id=scenario_id,
            out_dir=args.out,
            duration=args.duration,
            timestep=args.timestep,
            mode=args.mode,
            leak_scale=args.leak_scale,
        )
        if ok:
            success += 1

        if i % max(1, total // 20) == 0 or i == total:
            elapsed    = time.time() - t0
            avg_each   = elapsed / i
            remaining  = avg_each * (total - i)
            print(
                f"  [{i:>4}/{total}] Scenario-{scenario_id} | "
                f"elapsed={elapsed:.1f}s | avg={avg_each:.1f}s/scenario | "
                f"ETA={remaining:.0f}s | ok={success}"
            )

    elapsed_total = time.time() - t0
    print(f"\nDone. {success}/{total} scenarios saved to '{args.out}' in {elapsed_total:.1f}s")
    print(f"Each scenario: {args.duration // args.timestep} timestep rows × {len(junctions)} junctions")
    print(f"Approx total training rows: {success * (args.duration // args.timestep)}")


if __name__ == "__main__":
    main()
