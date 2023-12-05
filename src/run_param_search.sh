#!/bin/bash

#SBATCH --job-name=run_param_swp
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=20G
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL

#########################
### pre-configuration ###
#########################


### load conda environment ###

module load miniconda
conda init bash
source activate pytorch_envi

### run python script for sweep 

python3 $(pwd)/param_search.py > ver3_None.txt
