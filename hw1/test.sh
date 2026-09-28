#!/bin/bash

TARGET=${1:-hw1-1}
CPUS=${2:-4}
echo "Running tests with binary: ./$TARGET using $CPUS CPUs"

for i in {1..8}; do
    tc=$(printf "%02d" $i)
    echo "=== Testing t$tc ==="
    rm -f output.png
    time srun -N 1 -n 1 -c "$CPUS" ./$TARGET cases/hw1-1/t$tc.in.png output.png
    if [ -f output.png ]; then
        hw1-1-check cases/hw1-1/t$tc.out.png output.png
    else
        echo "Error: output.png was not generated!"
    fi
done
rm -f output.png
