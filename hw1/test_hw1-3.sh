#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=== Compiling hw1-3 ==="
make hw1-3

echo ""
echo "=== Running All 10 Test Cases (p01 ~ p10) ==="
printf "%-6s %-12s %-10s %-8s %-10s %-8s\n" "TC" "N / T" "Time" "Limit" "Cores" "Status"
printf "%s\n" "--------------------------------------------------------"

total_time=0
all_passed=true

for i in {1..10}; do
    tc=$(printf "%02d" $i)
    param_file="cases/hw1-3/p${tc}.txt"
    ans_file="cases/hw1-3/p${tc}.out"
    out_file="/tmp/out_p${tc}.txt"

    # 讀取參數
    N=$(grep "^N=" "$param_file" | cut -d= -f2 | tr -d '\r')
    T=$(grep "^T=" "$param_file" | cut -d= -f2 | tr -d '\r')
    seed=$(grep "^seed=" "$param_file" | cut -d= -f2 | tr -d '\r')
    theta=$(grep "^theta=" "$param_file" | cut -d= -f2 | tr -d '\r')
    theta=${theta:-0}
    cores=$(grep "^cores=" "$param_file" | cut -d= -f2 | tr -d '\r')
    cores=${cores:-8}
    timelimit=$(grep "^timelimit=" "$param_file" | cut -d= -f2 | tr -d '\r')
    timelimit=${timelimit:-10.0}

    # 計時執行
    t_start=$(date +%s%N)
    OMP_NUM_THREADS=$cores ./hw1-3 "$N" "$T" "$seed" "$theta" "$out_file"
    t_end=$(date +%s%N)

    elapsed=$(awk "BEGIN {printf \"%.3f\", ($t_end - $t_start)/1e9}")
    total_time=$(awk "BEGIN {printf \"%.3f\", $total_time + $elapsed}")

    # 驗證數值精準度 (能量 <= 1e-7, 抽樣溫度 <= 1e-9)
    check_result=$(python3 -c "
import sys
with open('$out_file') as f1, open('$ans_file') as f2:
    l1 = [x.strip() for x in f1.readlines()]
    l2 = [x.strip() for x in f2.readlines()]
if len(l1) != len(l2) or l1[0] != l2[0]:
    sys.exit(1)
e1, e2 = float(l1[1]), float(l2[1])
if abs(e1 - e2) / max(abs(e1), abs(e2), 1e-12) > 1e-7:
    sys.exit(2)
for i in range(2, len(l1)):
    v1, v2 = float(l1[i]), float(l2[i])
    if abs(v1 - v2) / max(abs(v1), abs(v2), 1e-12) > 1e-9:
        sys.exit(3)
sys.exit(0)
" 2>&1)
    status_code=$?

    if [ $status_code -eq 0 ]; then
        status="✅ PASS"
    else
        status="❌ FAIL"
        all_passed=false
    fi

    # 逾時判定警告
    is_tle=$(awk "BEGIN {print ($elapsed > $timelimit) ? 1 : 0}")
    if [ "$is_tle" -eq 1 ]; then
        status="${status} (TLE)"
        all_passed=false
    fi

    printf "%-6s %-12s %-10s %-8s %-10s %-8s\n" "p${tc}" "${N}/${T}" "${elapsed}s" "${timelimit}s" "$cores" "$status"
done

echo "--------------------------------------------------------"
if [ "$all_passed" = true ]; then
    echo "🎉 全部 10 個測資通過！總計耗時: ${total_time}s"
else
    echo "⚠️ 部分測資未通過或超時，總計耗時: ${total_time}s"
fi
