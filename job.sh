#!/usr/bin/env bash
#SBATCH --job-name=mpi
#SBATCH --output=mpi%j.out
#SBATCH --error=mpi%j.err
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --account=f202500009hpcvlabistul2x
#SBATCH --partition=normal-x86

srun mpi/src/docs-mpi tests/base/ex10000-1000000-100.in
