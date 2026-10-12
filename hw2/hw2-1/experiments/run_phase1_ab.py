#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import re
import statistics
import json
import glob

CASES = [
    ("06", "cases/hw2-1/06.txt", "cases/hw2-1/06.golden.bin"),
    ("09", "cases/hw2-1/09.txt", "cases/hw2-1/09.golden.bin"),
    ("10", "cases/hw2-1/10.txt", "cases/hw2-1/10.golden.bin"),
    ("12", "cases/hw2-1/12.txt", "cases/hw2-1/12.golden.bin"),
]

BIN_BASE = "./hw2-1-base-prof"
BIN_CLEAN = "./hw2-1-clean-rel"
TMP_DIR = "/tmp/phase1_ab"
os.makedirs(TMP_DIR, exist_ok=True)

SEQUENCE = ["BASE", "CLEAN", "CLEAN", "BASE", "BASE", "CLEAN", "CLEAN", "BASE", "BASE", "CLEAN"]

def cleanup_shm():
    for pattern in ["/dev/shm/osc_sm*", "/dev/shm/vader*"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass

def run_single(mode, case_txt, golden_bin, case_id, run_tag):
    cleanup_shm()
    bin_path = BIN_BASE if mode == "BASE" else BIN_CLEAN
    out_bin = os.path.join(TMP_DIR, f"out_{case_id}_{mode}_{run_tag}.bin")
    if os.path.exists(out_bin):
        os.remove(out_bin)
    
    cmd = [
        "mpirun",
        "-n", "8",
        bin_path,
        case_txt,
        out_bin
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall_sec = time.time() - t0
    
    if res.returncode != 0:
        print(f"[{case_id}][{mode}][{run_tag}] CRASH! code: {res.returncode}")
        print("STDERR:", res.stderr)
        print("STDOUT:", res.stdout)
        sys.exit(1)
        
    chk = subprocess.run(["hw2-1-check", out_bin, golden_bin], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if chk.returncode != 0:
        print(f"[{case_id}][{mode}][{run_tag}] Golden MISMATCH! {chk.stdout} {chk.stderr}")
        sys.exit(1)
        
    m = re.search(r"Compute time:\s*([0-9.]+)\s*s", res.stdout)
    if not m:
        print(f"[{case_id}][{mode}][{run_tag}] Parse failure:\n{res.stdout}")
        sys.exit(1)
    
    compute_time = float(m.group(1))
    return compute_time, wall_sec, out_bin

def stats(vals):
    mn = min(vals)
    mx = max(vals)
    med = statistics.median(vals)
    rng = mx - mn
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "runs": len(vals),
        "min": mn,
        "median": med,
        "max": mx,
        "range": rng,
        "mean": mean,
        "stdev": stdev,
        "raw": vals
    }

def main():
    print("=" * 65)
    print("Phase 1: Release Profile Removal A/B Benchmark (8 ranks)")
    print("BASE : hw2-1-base-prof (Unconditional MPI_Wtime & stats in SSSP)")
    print("CLEAN: hw2-1-clean-rel (Profile operations wrapped with #if ENABLE_PROFILE)")
    print("Sequence per case: Warm-up (Base, Clean) -> 10 valid interleaved runs")
    print("=" * 65, flush=True)

    all_results = {}

    for case_id, case_txt, golden_bin in CASES:
        print(f"\n>>> Starting Benchmark for Case {case_id} <<<", flush=True)
        
        # 1. Warm-up
        print("  [Warm-up] Running Base...", end="", flush=True)
        t_b_w, _, out_b_w = run_single("BASE", case_txt, golden_bin, case_id, "warmup")
        print(f" done ({t_b_w:.4f}s)", flush=True)

        print("  [Warm-up] Running Clean...", end="", flush=True)
        t_c_w, _, out_c_w = run_single("CLEAN", case_txt, golden_bin, case_id, "warmup")
        print(f" done ({t_c_w:.4f}s)", flush=True)

        chk_w = subprocess.run(["hw2-1-check", out_b_w, out_c_w], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if chk_w.returncode != 0:
            print("ERROR: Warm-up output mismatch!")
            sys.exit(1)
        print("  [Warm-up] Base and Clean byte-for-byte identical!", flush=True)
        os.remove(out_b_w)
        os.remove(out_c_w)

        # 2. Valid interleaved runs
        base_times = []
        clean_times = []
        last_base_out = None
        last_clean_out = None

        for idx, mode in enumerate(SEQUENCE, 1):
            print(f"  [Run {idx:02d}/10] Mode {mode}...", end="", flush=True)
            t_comp, _, out_f = run_single(mode, case_txt, golden_bin, case_id, f"run{idx}")
            print(f" done -> Compute time: {t_comp:.4f}s", flush=True)
            if mode == "BASE":
                base_times.append(t_comp)
                if last_base_out and os.path.exists(last_base_out):
                    os.remove(last_base_out)
                last_base_out = out_f
            else:
                clean_times.append(t_comp)
                if last_clean_out and os.path.exists(last_clean_out):
                    os.remove(last_clean_out)
                last_clean_out = out_f

        # Cross-comparison
        if last_base_out and last_clean_out:
            chk_cross = subprocess.run(["hw2-1-check", last_base_out, last_clean_out], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if chk_cross.returncode != 0:
                print(f"ERROR: Final Base vs Clean output mismatch for Case {case_id}!")
                sys.exit(1)
            print(f"  [Correctness] Case {case_id} Base vs Clean byte-for-byte MATCH!", flush=True)
            os.remove(last_base_out)
            os.remove(last_clean_out)

        base_st = stats(base_times)
        clean_st = stats(clean_times)
        diff_sec = base_st["median"] - clean_st["median"]
        imp_pct = (diff_sec / base_st["median"]) * 100.0

        all_results[case_id] = {
            "BASE": base_st,
            "CLEAN": clean_st,
            "diff_sec": diff_sec,
            "improvement_pct": imp_pct
        }

        print(f"\n--- Case {case_id} Results ---")
        print(f"  BASE  ({base_st['runs']} runs): Min={base_st['min']:.4f}s, Median={base_st['median']:.4f}s, Max={base_st['max']:.4f}s, Range={base_st['range']:.4f}s")
        print(f"  CLEAN ({clean_st['runs']} runs): Min={clean_st['min']:.4f}s, Median={clean_st['median']:.4f}s, Max={clean_st['max']:.4f}s, Range={clean_st['range']:.4f}s")
        print(f"  Median Improvement: {diff_sec:+.4f}s ({imp_pct:+.2f}%)", flush=True)

    # Full summary
    total_base_med = sum(all_results[c]["BASE"]["median"] for c in ["06", "09", "10", "12"])
    total_clean_med = sum(all_results[c]["CLEAN"]["median"] for c in ["06", "09", "10", "12"])
    total_diff = total_base_med - total_clean_med
    total_imp_pct = (total_diff / total_base_med) * 100.0

    summary = {
        "cases": all_results,
        "total_median": {
            "BASE": total_base_med,
            "CLEAN": total_clean_med,
            "diff_sec": total_diff,
            "improvement_pct": total_imp_pct
        }
    }

    with open("phase1_ab_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 65)
    print("PHASE 1 FINAL SUMMARY (Case 06, 09, 10, 12)")
    print("=" * 65)
    for c in ["06", "09", "10", "12"]:
        res = all_results[c]
        print(f"Case {c}: Base={res['BASE']['median']:.4f}s | Clean={res['CLEAN']['median']:.4f}s | Delta={res['diff_sec']:+.4f}s ({res['improvement_pct']:+.2f}%)")
    print("-" * 65)
    print(f"Total Medians: Base={total_base_med:.4f}s | Clean={total_clean_med:.4f}s | Net Improvement={total_diff:+.4f}s ({total_imp_pct:+.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()
