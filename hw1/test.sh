#!/bin/bash

TARGET=${1:-hw1-1}
CPUS=${2:-8}

for i in {1..8}; do
    tc=$(printf "%02d" $i)
    echo "=== Testing t$tc ==="
    time srun -N 1 -n 1 -c "$CPUS" ./$TARGET cases/$TARGET/t$tc.in.png output.png
    if [ -f output.png ]; then
        $TARGET-check cases/$TARGET/t$tc.out.png output.png
    fi
done
rm -f output.png
