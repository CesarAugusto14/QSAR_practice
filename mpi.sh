#!/bin/bash
#SBATCH --job-name=Scafoled_splitting
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition nocona
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
##SBATCH --mem-per-cpu=3994MB  ##3.9GB, Modify based on needs   

# module load gcc/10.1.0 openmpi/4.0.4
. $HOME/conda/etc/profile.d/conda.sh
conda activate torch_gpy
python test_scafold.py ./scaffold_data/train.csv ./scaffold_data/valid.csv 'swindow'
