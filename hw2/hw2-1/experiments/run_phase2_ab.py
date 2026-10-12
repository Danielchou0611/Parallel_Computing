#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import statistics
import json

BINARIES = {
    "legacy": "./hw2-1-chunk-legacy",
    "fixed64": "./hw2-1-chunk-fixed64",
    "adaptive": "./hw2-1-chunk-adaptive"
}

CASES = ["06", "09", "10"]

ROUNDS = [
    ["legacy", "fixed64", "adaptive"],  # Round 1
    ["adaptive", "fixed64", "legacy"],  # Round 2
    ["fixed64", "legacy", "adaptive"],  # Round 3
    ["legacy", "adaptive", "fixed64"],  # Round 4
    ["fixed64", "adaptive", "legacy"]   # Round 5
]

TMP_DIR = "/dev/shm/phase2_bench"
os.makedirs(TMP_DIR, exist_ok=True)

def parse_compute_time(output):
    m = re.search(r"Compute time:\s*([0-9.]+)\s*s", output)
    if m:
        return float(m.group(1))
    return None

results = {c: {"legacy": [], "fixed64": [], "adaptive": []} for c in CASES}

print("=================================================================", flush=True)
print("  PHASE 2: FRONTIER CHUNK STRATEGY 3-WAY INTERLEAVED A/B BENCHMARK", flush=True)
print("=================================================================", flush=True)

for case_id in CASES:
    case_file = f"cases/hw2-1/{case_id}.txt"
    print(f"\n>>> Starting Case {case_id} ({case_file})", flush=True)

    # 1. Warm-up
    print(f"--- Warm-up for Case {case_id} ---", flush=True)
    for mode in ["legacy", "fixed64", "adaptive"]:
        bin_path = BINARIES[mode]
        out_bin = f"{TMP_DIR}/warmup_{mode}_{case_id}.bin"
        cmd = ["mpirun", "-n", "8", bin_path, case_file, out_bin]
        res = subprocess.run(cmd, capture_output=True, text=True)
        t = parse_compute_time(res.stdout)
        print(f"  [Warm-up] {mode:8s}: {t:.4f} s" if t else f"  [Warm-up] {mode:8s}: FAILED", flush=True)
        if os.path.exists(out_bin):
            os.remove(out_bin)

    # 2. 5 Interleaved Rounds
    print(f"--- 5 Interleaved Rounds for Case {case_id} ---", flush=True)
    for r_idx, order in enumerate(ROUNDS, 1):
        print(f"  Round {r_idx} Order: {' -> '.join(order)}", flush=True)
        for mode in order:
            bin_path = BINARIES[mode]
            out_bin = f"{TMP_DIR}/r{r_idx}_{mode}_{case_id}.bin"
            cmd = ["mpirun", "-n", "8", bin_path, case_file, out_bin]
            res = subprocess.run(cmd, capture_output=True, text=True)
            t = parse_compute_time(res.stdout)
            if t is None:
                print(f"    ERROR running {mode} on Case {case_id}! stderr:\n{res.stderr}", flush=True)
                sys.exit(1)
            results[case_id][mode].append(t)
            print(f"    {mode:8s}: {t:.4f} s", flush=True)
            if os.path.exists(out_bin):
                os.remove(out_bin)

    print(f"\n--- Summary for Case {case_id} ---", flush=True)
    for mode in ["legacy", "fixed64", "adaptive"]:
        samples = results[case_id][mode]
        s_min = min(samples)
        s_med = statistics.median(samples)
        s_max = max(samples)
        s_range = s_max - s_min
        s_mean = statistics.mean(samples)
        print(f"  {mode:8s} (N={len(samples)}): min={s_min:.4f} med={s_med:.4f} max={s_max:.4f} range={s_range:.4f} mean={s_mean:.4f} s", flush=True)
        print(f"    samples: {[round(x, 4) for x in samples]}", flush=True)

# Final Overall Summary
print("\n" + "="*65, flush=True)
print("  PHASE 2 FINAL STATISTICAL ANALYSIS", flush=True)
print("="*65, flush=True)

stats_summary = {}
for case_id in CASES:
    stats_summary[case_id] = {}
    legacy_med = statistics.median(results[case_id]["legacy"])
    print(f"\nCase {case_id}:", flush=True)
    for mode in ["legacy", "fixed64", "adaptive"]:
        samples = results[case_id][mode]
        med = statistics.median(samples)
        diff_legacy = med - legacy_med
        pct_legacy = (diff_legacy / legacy_med) * 100.0
        stats_summary[case_id][mode] = {
            "samples": samples,
            "min": min(samples),
            "median": med,
            "max": max(samples),
            "range": max(samples) - min(samples),
            "mean": statistics.mean(samples),
            "diff_vs_legacy": diff_legacy,
            "pct_vs_legacy": pct_legacy
        }
        print(f"  {mode:8s}: Median = {med:8.4f} s | vs Legacy: {diff_legacy:+8.4f} s ({pct_legacy:+6.2f}%) | Min={min(samples):.4f} Max={max(samples):.4f}", flush=True)

print("\n-----------------------------------------------------------------", flush=True)
print("  TOTAL MEDIAN SUM COMPARISON", flush=True)
print("-----------------------------------------------------------------", flush=True)

# Median Sum across affected cases (Case 09, 10, or all 3)
for group_name, group_cases in [("All 3 Cases (06, 09, 10)", ["06", "09", "10"]), ("Cooperative Cases (09, 10)", ["09", "10"])]:
    print(f"\n{group_name}:", flush=True)
    leg_sum = sum(stats_summary[c]["legacy"]["median"] for c in group_cases)
    fix_sum = sum(stats_summary[c]["fixed64"]["median"] for c in group_cases)
    adp_sum = sum(stats_summary[c]["adaptive"]["median"] for c in group_cases)

    print(f"  Legacy   Median Sum: {leg_sum:.4f} s (Baseline)", flush=True)
    print(f"  Fixed64  Median Sum: {fix_sum:.4f} s (diff: {fix_sum - leg_sum:+.4f} s, {(fix_sum - leg_sum)/leg_sum * 100:+.2f}%)", flush=True)
    print(f"  Adaptive Median Sum: {adp_sum:.4f} s (diff: {adp_sum - leg_sum:+.4f} s, {(adp_sum - leg_sum)/leg_sum * 100:+.2f}%)", flush=True)

with open("phase2_ab_results.json", "w") as f:
    json.dump({"results": results, "stats": stats_summary}, f, indent=2)

print("\nBenchmark completed. Results written to phase2_ab_results.json", flush=True)
