#!/bin/bash

TARGET=${1:-hw1-1}
CPUS=${2:-8}

echo "=== Benchmark test for ./$TARGET (using $CPUS cores) ==="
printf "%-6s  %-6s  %-8s\n" "TC" "STAT" "TIME"
echo "──────────────────────"

TOTAL_TIME=0.0

for i in {1..8}; do
    tc=$(printf "%02d" $i)
    inp="cases/hw1-1/t$tc.in.png"
    ref="cases/hw1-1/t$tc.out.png"
    out="output.png"
    rm -f "$out"

    # Execute and measure pure runtime inside the compute node
    DUR=$(srun -N 1 -n 1 -c "$CPUS" python3 -c "
import time, subprocess
t0 = time.time()
res = subprocess.run(['./$TARGET', '$inp', '$out'], stdout=subprocess.DEVNULL)
t1 = time.time()
print(f'{t1 - t0:.3f}')
")

    # Check accuracy with hw1-1-check
    MATCH_STR=$(hw1-1-check "$ref" "$out" 2>&1)
    if [[ "$MATCH_STR" =~ "100.00%" ]] || [[ "$MATCH_STR" =~ "need 99%" ]]; then
        STAT="AC"
    else
        STAT="WA"
    fi

    printf "%-6s  %-6s  %6ss\n" "t$tc" "$STAT" "$DUR"
    TOTAL_TIME=$(python3 -c "print(f'{$TOTAL_TIME + float(\"$DUR\"):.3f}')")
    rm -f "$out"
done

echo "──────────────────────"
printf "Total:            %6ss\n" "$TOTAL_TIME"
