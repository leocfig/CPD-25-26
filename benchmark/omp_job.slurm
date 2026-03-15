#!/bin/bash
#SBATCH --job-name=omp_bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12

LOCAL_DIR="/tmp/$USER/omp_bench_$SLURM_JOB_ID"
mkdir -p $LOCAL_DIR

echo "Starting job on node: $SLURMD_NODENAME"
echo "Working in local directory: $LOCAL_DIR"

cp -r $SLURM_SUBMIT_DIR/omp $LOCAL_DIR/
cp -r $SLURM_SUBMIT_DIR/serial $LOCAL_DIR/
cp -r $SLURM_SUBMIT_DIR/tests $LOCAL_DIR/
cp -r $SLURM_SUBMIT_DIR/benchmarks $LOCAL_DIR/

cd $LOCAL_DIR

(cd serial/src && make clean && make)
(cd omp/src && make clean && make)

python3 benchmark/omp_bench.py \
    -s ./serial/src/docs \
    -p ./omp/src/docs-omp \
    -i ./tests/base \
    -o results_${SLURMD_NODENAME}.csv \
    -t 1 2 4 6

echo "Copying results back to AFS..."
cp results_${SLURMD_NODENAME}.csv $SLURM_SUBMIT_DIR/

cd $SLURM_SUBMIT_DIR
rm -rf $LOCAL_DIR

echo "Job finished successfully on $SLURMD_NODENAME!"