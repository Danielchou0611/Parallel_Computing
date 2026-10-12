#!/usr/bin/env python3
import os
import sys
import subprocess
import time

CASES = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"]
TMP_DIR = "/dev/shm/verify_hybrid"
os.makedirs(TMP_DIR, exist_ok=True)

print("=" * 70)
print("  24-RANK FULL CORRECTNESS REGRESSION: HYBRID (MODE 3) CLEAN RELEASE")
print("=" * 70)

all_passed = True
for c in CASES:
    case_txt = f"cases/hw2-1/{c}.txt"
    golden_bin = f"cases/hw2-1/{c}.golden.bin"
    out_f = f"{TMP_DIR}/hybrid_{c}.bin"

    print(f"Testing Case {c:2s} (24 ranks)... ", end="", flush=True)
    t0 = time.time()
    cmd = ["mpirun", "--use-hwthread-cpus", "-n", "24", "./hw2-1", case_txt, out_f]
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"FAILED (Crash / Deadlock / Error)! Return code: {res.returncode}")
        print(f"stderr:\n{res.stderr}")
        all_passed = False
        break

    chk = subprocess.run(["hw2-1-check", out_f, golden_bin], capture_output=True, text=True)
    if chk.returncode != 0:
        print(f"FAILED (Golden Mismatch)! {chk.stdout.strip()} {chk.stderr.strip()}")
        all_passed = False
        break

    print(f"PASS [MATCH Golden] ({elapsed:.2f}s)")
    if os.path.exists(out_f):
        os.remove(out_f)

print("=" * 70)
if all_passed:
    print("ALL 17 CASES 100% BYTE-FOR-BYTE MATCH GOLDEN ON 24 RANKS!")
else:
    print("REGRESSION FAILED!")
    sys.exit(1)
print("=" * 70)
