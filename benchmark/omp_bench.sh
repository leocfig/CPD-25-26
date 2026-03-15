#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

AFS_DIR=$(pwd)
NODE_NAME=$(hostname)
# Use the process ID ($$) to ensure a unique temp folder
LOCAL_DIR="/tmp/$USER/omp_bench_${NODE_NAME}_$$"

echo "================================================="
echo "Starting Manual Benchmark on Node: $NODE_NAME"
echo "Project AFS Directory: $AFS_DIR"
echo "Local Temp Workspace:   $LOCAL_DIR"
echo "================================================="

mkdir -p "$LOCAL_DIR"

(cd ./omp/src && make clean && make)
(cd ./serial/src && make clean && make)

echo "Copying project to local SSD..."
cp "$AFS_DIR/benchmark/omp_bench.py" "$LOCAL_DIR/bench.py"
cp "$AFS_DIR/omp/src/docs-omp" "$LOCAL_DIR/"
cp "$AFS_DIR/serial/src/docs" "$LOCAL_DIR/"
cp -r "$AFS_DIR/tests/base" "$LOCAL_DIR/tests"

cd "$LOCAL_DIR"

echo "Starting Python benchmark suite..."
python3 bench.py \
    -s ./docs \
    -p ./docs-omp \
    -i ./tests \
    -o "results_${NODE_NAME}.csv" \
    -t 1 2 4 6

echo "Copying results back to AFS..."
cp "results_${NODE_NAME}.csv" "$AFS_DIR/"

# 8. Clean up the node's local drive so we don't leave trash behind
echo "Cleaning up temp workspace..."
cd "$AFS_DIR"
rm -rf "$LOCAL_DIR"

echo "================================================="
echo "Success! Results saved as results_${NODE_NAME}.csv"
echo "================================================="
