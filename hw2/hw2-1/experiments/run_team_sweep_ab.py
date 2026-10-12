#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import statistics
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ranks", type=int, default=8, help="Number of MPI processes")
parser.add_argument("--cases", nargs="+", default=["09", "10"], help="Cases to test")
parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
args = parser.parse_args()

P = args.ranks
CASES = args.cases
NUM_ROUNDS = args.rounds

BINARIES = {
    "mode0": "./hw2-1-team-mode0",
    "mode1": "./hw2-1-team-mode1",
    "mode2": "./hw2-1-team-mode2"
}

PROF_BINARIES = {
    "mode0": "./hw2-1-team-mode0-prof",
    "mode1": "./hw2-1-team-mode1-prof",
    "mode2": "./hw2-1-team-mode2-prof"
}

ROUNDS_ORDER = [
    ["mode0", "mode1", "mode2"],
    ["mode2", "mode1", "mode0"],
    ["mode1", "mode0", "mode2"],
    ["mode0", "mode2", "mode1"],
    ["mode1", "mode2", "mode0"]
]

TMP_DIR = f"/dev/shm/team_bench_{P}"
os.makedirs(TMP_DIR, exist_ok=True)

def parse_compute_time(output):
    m = re.search(r"Compute time:\s*([0-9.]+)\s*s", output)
    if m:
        return float(m.group(1))
    return None

def parse_profile(output):
    # [Profile] Alloc: 0.0004s | Pass1: 0.6953s | Prefix: 0.0763s | Pass2: 0.7303s | (Build: 1.5023s) | SSSP: 7.6432s | Copy: 0.0002s
    # [SSSP Stats] Rounds: 320 | Collectives: 0.2814s | Relax: 7.2145s | TotalFrontier: 26843545
    # [Team 0 Source 0 (v=8415300)] SSSP Time: 3.8210s
    res = {}
    m_p = re.search(r"SSSP:\s*([0-9.]+)\s*s", output)
    if m_p: res["sssp_total"] = float(m_p.group(1))
    m_c = re.search(r"Collectives:\s*([0-9.]+)\s*s", output)
    if m_c: res["collective_time"] = float(m_c.group(1))
    m_r = re.search(r"Relax:\s*([0-9.]+)\s*s", output)
    if m_r: res["relax_time"] = float(m_r.group(1))
    m_rnd = re.search(r"Rounds:\s*([0-9]+)", output)
    if m_rnd: res["rounds"] = int(m_rnd.group(1))
    
    src_times = re.findall(r"\[Team\s+(\d+)\s+Source\s+(\d+)[^\]]*\]\s+SSSP Time:\s*([0-9.]+)\s*s", output)
    res["sources"] = [(int(t), int(s), float(tm)) for t, s, tm in src_times]
    return res

results = {c: {"mode0": [], "mode1": [], "mode2": []} for c in CASES}
profile_data = {c: {} for c in CASES}

print("="*70, flush=True)
print(f"  SOURCE CONCURRENCY SWEEP BENCHMARK (P = {P} Ranks)", flush=True)
print("="*70, flush=True)

for case_id in CASES:
    case_file = f"cases/hw2-1/{case_id}.txt"
    print(f"\n>>> Starting Case {case_id} (P={P})", flush=True)

    # 1. Warm-up
    print(f"--- Warm-up for Case {case_id} ---", flush=True)
    for m in ["mode0", "mode1", "mode2"]:
        out_bin = f"{TMP_DIR}/warmup_{m}_{case_id}.bin"
        res = subprocess.run(["mpirun", "--use-hwthread-cpus", "-n", str(P), BINARIES[m], case_file, out_bin], capture_output=True, text=True)
        t = parse_compute_time(res.stdout)
        print(f"  [Warm-up] {m}: {t:.4f} s" if t else f"  [Warm-up] {m}: FAILED", flush=True)
        if os.path.exists(out_bin): os.remove(out_bin)

    # 2. 5 Interleaved Rounds
    print(f"--- {NUM_ROUNDS} Interleaved Rounds for Case {case_id} ---", flush=True)
    for r_idx in range(1, NUM_ROUNDS + 1):
        order = ROUNDS_ORDER[(r_idx - 1) % len(ROUNDS_ORDER)]
        print(f"  Round {r_idx} Order: {' -> '.join(order)}", flush=True)
        for m in order:
            out_bin = f"{TMP_DIR}/r{r_idx}_{m}_{case_id}.bin"
            res = subprocess.run(["mpirun", "--use-hwthread-cpus", "-n", str(P), BINARIES[m], case_file, out_bin], capture_output=True, text=True)
            t = parse_compute_time(res.stdout)
            if t is None:
                print(f"    ERROR running {m}! stderr:\n{res.stderr}", flush=True)
                sys.exit(1)
            results[case_id][m].append(t)
            print(f"    {m}: {t:.4f} s", flush=True)
            if os.path.exists(out_bin): os.remove(out_bin)

    # 3. Profile Diagnostic Run
    print(f"--- Diagnostic Profiling for Case {case_id} ---", flush=True)
    for m in ["mode0", "mode1", "mode2"]:
        out_bin = f"{TMP_DIR}/prof_{m}_{case_id}.bin"
        res = subprocess.run(["mpirun", "--use-hwthread-cpus", "-n", str(P), PROF_BINARIES[m], case_file, out_bin], capture_output=True, text=True)
        p_info = parse_profile(res.stdout)
        profile_data[case_id][m] = p_info
        print(f"  [{m}] SSSP Total: {p_info.get('sssp_total', 0):.4f}s | Collectives: {p_info.get('collective_time', 0):.4f}s | Relax: {p_info.get('relax_time', 0):.4f}s", flush=True)
        for t_id, s_id, s_tm in p_info.get("sources", []):
            print(f"       Team {t_id} Source {s_id}: {s_tm:.4f}s", flush=True)
        if os.path.exists(out_bin): os.remove(out_bin)

# Summary
print("\n" + "="*70, flush=True)
print(f"  FINAL SUMMARY (P = {P} Ranks)", flush=True)
print("="*70, flush=True)

summary_stats = {}
for c in CASES:
    summary_stats[c] = {}
    base_med = statistics.median(results[c]["mode0"])
    print(f"\nCase {c}:", flush=True)
    for m in ["mode0", "mode1", "mode2"]:
        s = results[c][m]
        med = statistics.median(s)
        diff = med - base_med
        pct = (diff / base_med) * 100.0
        summary_stats[c][m] = {
            "samples": s, "min": min(s), "median": med, "max": max(s), "range": max(s)-min(s),
            "mean": statistics.mean(s), "diff": diff, "pct": pct
        }
        print(f"  {m:8s}: Median={med:8.4f}s | vs Mode0: {diff:+8.4f}s ({pct:+6.2f}%) | Min={min(s):.4f}s Max={max(s):.4f}s Range={max(s)-min(s):.4f}s", flush=True)

out_json = f"team_sweep_P{P}_results.json"
with open(out_json, "w") as f:
    json.dump({"ranks": P, "results": results, "stats": summary_stats, "profile": profile_data}, f, indent=2)

print(f"\nResults saved to {out_json}", flush=True)
