#!/usr/bin/env python3
import os
import sys
import subprocess

CASES = ["01", "06", "09", "10", "14", "16", "17"]
MODES = [
    ("legacy", "./hw2-1-chunk-legacy"),
    ("fixed64", "./hw2-1-chunk-fixed64"),
    ("adaptive", "./hw2-1-chunk-adaptive")
]
TMP_DIR = "/dev/shm/phase2_correctness"
os.makedirs(TMP_DIR, exist_ok=True)

for c in CASES:
    case_txt = f"cases/hw2-1/{c}.txt"
    golden_bin = f"cases/hw2-1/{c}.golden.bin"
    out_files = {}
    
    print(f"Testing Case {c}...", end="", flush=True)
    for tag, bin_path in MODES:
        out_f = f"{TMP_DIR}/{tag}_{c}.bin"
        out_files[tag] = out_f
        res = subprocess.run(["mpirun", "-n", "8", bin_path, case_txt, out_f], capture_output=True, text=True)
        if res.returncode != 0:
            print(f" CRASH ({tag})! {res.stderr}")
            sys.exit(1)
        chk_g = subprocess.run(["hw2-1-check", out_f, golden_bin], capture_output=True, text=True)
        if chk_g.returncode != 0:
            print(f" Golden MISMATCH ({tag})! {chk_g.stdout}")
            sys.exit(1)
            
    # Cross check
    chk_lf = subprocess.run(["hw2-1-check", out_files["legacy"], out_files["fixed64"]], capture_output=True, text=True)
    chk_la = subprocess.run(["hw2-1-check", out_files["legacy"], out_files["adaptive"]], capture_output=True, text=True)
    if chk_lf.returncode != 0 or chk_la.returncode != 0:
        print(" Cross-mode MISMATCH!")
        sys.exit(1)
        
    for f in out_files.values():
        os.remove(f)
    print(" PASS (All 3 MATCH golden & MATCH each other!)")

print("\nALL 7 CASES 100% BYTE-FOR-BYTE MATCH ACROSS ALL 3 CHUNK MODES!")
