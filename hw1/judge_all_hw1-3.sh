#!/bin/bash

# If argument provided (e.g. p01), judge single case.
# Otherwise, judge all 10 cases to submit and record onto the official leaderboard.
if [ -n "$1" ]; then
    echo "=== Judging Single Case: $1 ==="
    hw1-3-judge "$1"
else
    echo "=== Judging All 10 Cases & Submitting to Leaderboard ==="
    hw1-3-judge
fi

