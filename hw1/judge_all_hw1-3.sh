#!/bin/bash

echo "=== Judging HW1-3: p01 ~ p10 ==="
for i in {1..10}; do
    tc=$(printf "%02d" $i)
    hw1-3-judge "p$tc"
done
