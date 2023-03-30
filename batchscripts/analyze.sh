#!/bin/bash
#SBATCH --job-name=analyze_val_all-chars
#SBATCH --output=outputs/analyze_val_all-chars.txt
#SBATCH --mail-user=mai@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --qos=batch
##SBATCH --gres=gpu:1


###############################################################################
#             ! MODIFY BOTH OUTPUT FILE AND VARIABLES BELOW !                 #
###############################################################################

# TRAINING PARAMETERS

# ENV VARIABLES


# LOGGING SETUP

# export PYTHONPATH=${PYTHONPATH}:${PAPER_REPO}
# PYTHONPATH=${PAPER_REPO}

python3 analyze.py --attacks_file "data/attacks_val_all-chars.json"
