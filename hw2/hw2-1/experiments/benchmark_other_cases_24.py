#!/usr/bin/env python3
import subprocess
import re
import statistics

CASES = ["04", "06", "07", "08", "11", "13"]
MODES = ["mode0", "mode1", "mode2"]
BINARIES = {
    "mode0": "./hw2-1-team-mode0",
    "mode1": "./hw2-1-team-mode1",
    "mode2": "./hw2-1-team-mode2"
}
TMP_DIR = "/dev/shm/other_cases_24"
import os; os.makedirs(TMP_DIR, exist_ok=True)

def parse_time(out):
    m = re.search(r"Compute time:\s*([0-9.]+)\s*s", out)
    return float(m.group(1)) if m else None

results = {c: {m: [] for m in MODES} for c in CASES}

for c in CASES:
    case_txt = f"cases/hw2-1/{c}.txt"
    print(f"Testing Case {c} (24 ranks)...", flush=True)
    for m in MODES:
        for _ in range(3):
            out_bin = f"{TMP_DIR}/{m}_{c}.bin"
            res = subprocess.run(["mpirun", "--use-hwthread-cpus", "-n", "24", BINARIES[m], case_txt, out_bin], capture_output=True, text=True)
            t = parse_time(res.stdout)
            if t: results[c][m].append(t)
            if os.path.exists(out_bin): os.remove(out_bin)
    
    m0_med = statistics.median(results[c]["mode0"])
    m1_med = statistics.median(results[c]["mode1"])
    m2_med = statistics.median(results[c]["mode2"])
    print(f"  Case {c}: Mode0={m0_med:.4f}s | Mode1={m1_med:.4f}s ({(m1_med-m0_med)/m0_med*100:+.2f}%) | Mode2={m2_med:.4f}s ({(m2_med-m0_med)/m0_med*100:+.2f}%)", flush=True)
