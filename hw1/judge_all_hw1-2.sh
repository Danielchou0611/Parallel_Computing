#!/bin/bash

echo "=== Judging HW1-2: a01 ~ a08 ==="
for i in {1..8}; do
    tc=$(printf "%02d" $i)
    hw1-2-judge "a$tc"
done

echo "=== Judging HW1-2: b01 ~ b08 ==="
for i in {1..8}; do
    tc=$(printf "%02d" $i)
    hw1-2-judge "b$tc"
done
