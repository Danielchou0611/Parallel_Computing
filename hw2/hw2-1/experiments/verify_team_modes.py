#!/usr/bin/env python3
import os
import sys
import subprocess

CASES = ["01", "07", "09", "10", "14", "16", "17"]
MODES = [
    ("mode0", "./hw2-1-team-mode0"),
    ("mode1", "./hw2-1-team-mode1"),
    ("mode2", "./hw2-1-team-mode2")
]
TMP_DIR = "/dev/shm/team_correctness"
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
    chk_01 = subprocess.run(["hw2-1-check", out_files["mode0"], out_files["mode1"]], capture_output=True, text=True)
    chk_02 = subprocess.run(["hw2-1-check", out_files["mode0"], out_files["mode2"]], capture_output=True, text=True)
    if chk_01.returncode != 0 or chk_02.returncode != 0:
        print(" Cross-mode MISMATCH!")
        sys.exit(1)

    for f in out_files.values():
        os.remove(f)
    print(" PASS (All 3 modes MATCH golden & MATCH each other!)")

print("\nALL 7 CASES 100% BYTE-FOR-BYTE MATCH ACROSS ALL 3 SOURCE_TEAM_MODES!")
