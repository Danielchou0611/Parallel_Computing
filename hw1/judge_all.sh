#!/bin/bash

echo "=== Judging t01 to t08 individually (No Cooldown) ==="
printf "  %-4s %-6s %-6s %-6s\n" "TC" "STAT" "NEW" "BEST"
echo "────────────────────────────────────"

TOTAL_NEW=0.0
TOTAL_BEST=0.0

for i in {1..8}; do
    tc=$(printf "%02d" $i)
    OUTPUT=$(hw1-1-judge "t$tc" 2>&1)
    
    # Extract line like "t01  AC     0.01 0.02 ↓"
    LINE=$(echo "$OUTPUT" | grep -E "t$tc\s+(AC|WA|TLE|RE|TLE\+)")
    if [ -n "$LINE" ]; then
        echo "  $LINE"
        NEW_TIME=$(echo "$LINE" | awk '{print $3}')
        BEST_TIME=$(echo "$LINE" | awk '{print $4}')
        TOTAL_NEW=$(python3 -c "print(f'{$TOTAL_NEW + float(\"$NEW_TIME\"):.2f}')" 2>/dev/null || echo "$TOTAL_NEW")
        TOTAL_BEST=$(python3 -c "print(f'{$TOTAL_BEST + float(\"$BEST_TIME\"):.2f}')" 2>/dev/null || echo "$TOTAL_BEST")
    else
        echo "  t$tc: judge failed or output format changed"
        echo "$OUTPUT"
    fi
done

echo "────────────────────────────────────"
printf "Estimated Total:  %6ss (BEST was %6ss)\n" "$TOTAL_NEW" "$TOTAL_BEST"
echo ""
echo "Note: Single-case runs are exempt from the 10-min cooldown and do not update the scoreboard."
echo "If you want to officially record this score to the leaderboard, run: hw1-1-judge"
