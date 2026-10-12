#!/usr/bin/env python3
import os
import sys
import subprocess

CASES = ["01", "06", "09", "10", "12", "14", "16", "17"]
BASE_BIN = "./hw2-1-base-prof"
CLEAN_BIN = "./hw2-1-clean-rel"
TMP_DIR = "/tmp/phase1_correctness"
os.makedirs(TMP_DIR, exist_ok=True)

for c in CASES:
    case_txt = f"cases/hw2-1/{c}.txt"
    golden_bin = f"cases/hw2-1/{c}.golden.bin"
    out_base = f"{TMP_DIR}/base_{c}.bin"
    out_clean = f"{TMP_DIR}/clean_{c}.bin"
    
    print(f"Testing Case {c}...", end="", flush=True)
    
    # Run base
    res_b = subprocess.run(["mpirun", "-n", "8", BASE_BIN, case_txt, out_base], capture_output=True, text=True)
    if res_b.returncode != 0:
        print(f" BASE CRASH! {res_b.stderr}")
        sys.exit(1)
        
    # Run clean
    res_c = subprocess.run(["mpirun", "-n", "8", CLEAN_BIN, case_txt, out_clean], capture_output=True, text=True)
    if res_c.returncode != 0:
        print(f" CLEAN CRASH! {res_c.stderr}")
        sys.exit(1)
        
    # Check clean vs golden
    chk_g = subprocess.run(["hw2-1-check", out_clean, golden_bin], capture_output=True, text=True)
    if chk_g.returncode != 0:
        print(f" CLEAN vs GOLDEN MISMATCH! {chk_g.stdout} {chk_g.stderr}")
        sys.exit(1)
        
    # Check clean vs base
    chk_b = subprocess.run(["hw2-1-check", out_clean, out_base], capture_output=True, text=True)
    if chk_b.returncode != 0:
        print(f" CLEAN vs BASE MISMATCH! {chk_b.stdout} {chk_b.stderr}")
        sys.exit(1)
        
    os.remove(out_base)
    os.remove(out_clean)
    print(" PASS (Golden MATCH & Base MATCH!)")

print("\nALL 8 CASES VERIFIED SUCCESSFULLY: 100% BYTE-FOR-BYTE MATCH!")
