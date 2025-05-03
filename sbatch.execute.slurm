#!/bin/bash
#SBATCH --job-name=test-apptainer
#SBATCH --time=0:03:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# TODO - change to run apptainer
# Path to your SIF file
CONTAINER="aml_container.sif"

# Run command inside the container
srun /usr/bin/apptainer exec --nv "$CONTAINER" python -c "import numpy as np; print(np.random.choice(5,3))"
