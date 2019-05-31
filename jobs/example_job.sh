#!/bin/bash
#SBATCH -N 1
#SBATCH -t 120:00:00

#SBATCH --job-name=JOB_NAME
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=YOUR_MAIL

# Load modules.
module load Miniconda3/4.3.27

# Load venv, execute script.
source activate YOUR_VENV

# Train stuff.
python3 -m param --dist norm --dist-args -10 10 -3 3 0.5 3 --gen-mode --n-samples 100000 --grid-size 1000 --epochs 50 --dense-layers 40 --depth 25 --kernels 15 --kernel-size 41 --name gen1000-100000 &
python3 -m param --dist norm --dist-args -10 10 -3 3 0.5 3 --gen-mode --n-samples 20000 --grid-size 1000 --epochs 50 --dense-layers 40 --depth 25 --kernels 15 --kernel-size 41 --name gen1000-20000 &
python3 -m param --dist norm --dist-args -10 10 -3 3 0.5 3 --n-samples 1000000 --grid-size 1000 --epochs 50 --dense-layers 40 --depth 25 --kernels 15 --kernel-size 41 --name 1000-1000000 &
python3 -m param --dist norm --dist-args -10 10 -3 3 0.5 3 --gen-mode --n-samples 20000 --grid-size 1000 --batch-size 32 --epochs 50 --dense-layers 40 --depth 25 --kernels 15 --kernel-size 41 --name gs1000-2000$

# Deactivate venv.
source deactivate

# Wait until done before ending the job.
wait

