#!/bin/bash

TARGET=${1:-hw1-2}

if [ "$TARGET" = "hw1-1" ]; then
    for i in {1..8}; do
        tc=$(printf "%02d" $i)
        hw1-1-judge "t$tc"
    done
elif [ "$TARGET" = "hw1-2" ]; then
    for i in {1..8}; do
        tc=$(printf "%02d" $i)
        hw1-2-judge "a$tc"
    done
    for i in {1..8}; do
        tc=$(printf "%02d" $i)
        hw1-2-judge "b$tc"
    done
elif [ "$TARGET" = "hw1-3" ]; then
    for i in {1..10}; do
        tc=$(printf "%02d" $i)
        hw1-3-judge "p$tc"
    done
else
    echo "Unknown target: $TARGET (use hw1-1, hw1-2, or hw1-3)"
fi
