#!/bin/bash

TARGET=${1:-hw1-1}

for i in {1..8}; do
    tc=$(printf "%02d" $i)
    $TARGET-judge "t$tc"
done
