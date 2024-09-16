#!/bin/bash
#SBATCH --partition=nocona       # Partition name
#SBATCH --job-name=test_scaffold # Job name
#SBATCH --output=out_%j.log      # Standard output log file (%j = job ID)
#SBATCH --error=err_%j.log       # Error log file (%j = job ID)
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=2        # Number of CPU cores per task
#SBATCH --mem=4G                 # Memory per node
#SBATCH --time=048:00:00          # Time limit
#SBATCH --nodes=1                # Number of nodes
python test_scafold.py ./scaffold_data/train.csv ./scaffold_data/valid.csv 'swindow'