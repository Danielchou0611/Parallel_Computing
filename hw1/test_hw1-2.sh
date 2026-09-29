#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=== Compiling hw1-2 ==="
make hw1-2

CASES=(
  a01 a02 a03 a04 a05 a06 a07 a08
  b01 b02 b03 b04 b05 b06 b07 b08
)

echo "=== Running All 16 Test Cases ==="
total_time=0

for tc in "${CASES[@]}"; do
    imgA="cases/hw1-2/${tc}.in_A.png"
    imgB="cases/hw1-2/${tc}.in_B.png"
    ans="cases/hw1-2/${tc}.out.txt"
    out="/tmp/out_${tc}.txt"

    # 測量執行時間
    t_start=$(date +%s%N)
    ./hw1-2 "$imgA" "$imgB" "$out"
    t_end=$(date +%s%N)

    elapsed=$(awk "BEGIN {printf \"%.3f\", ($t_end - $t_start)/1e9}")
    total_time=$(awk "BEGIN {printf \"%.3f\", $total_time + $elapsed}")

    # 比對輸出是否一致
    if diff -q "$ans" "$out" > /dev/null 2>&1; then
        echo "[$tc] ✅ PASS (${elapsed}s)"
    else
        echo "[$tc] ❌ FAIL (Output mismatch!)"
        diff -u "$ans" "$out" | head -n 10
        exit 1
    fi
done

echo "================================="
echo "🎉 全部通過！總耗時: ${total_time}s"
echo "================================="
