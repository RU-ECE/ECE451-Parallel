#!/bin/bash
# submit this job by giving the following command:
# sbatch gol.slurm
#SBATCH --job-name=gol
#SBATCH --output=gol.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

# Load any necessary modules (if required)
# module load openmpi

# Run the compiled program
srun  --mpi=openmpi  ./gol
