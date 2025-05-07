#!/bin/bash
#SBATCH --job-name=aml_training
#SBATCH --partition=st,general
#SBATCH --qos=short             # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=1:00:00          # Request run time. Default is 1 minute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000             # Request GB of RAM
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END         # Set mail type to 'END' to receive a mail when the job finishes.

# Path to your SIF file
CONTAINER="aml_container.sif"

# Run command inside the container
srun /usr/bin/apptainer run --nv $CONTAINER --model pna --data Small_HI --n_epochs 2 --output_dir /tmp/results