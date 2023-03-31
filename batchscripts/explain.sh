#!/bin/bash
#SBATCH --job-name=explain_val_no-letters
#SBATCH --output=outputs/explain_val_no-letters.txt
#SBATCH --mail-user=mai@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --qos=batch
#SBATCH --gres=gpu:1


###############################################################################
#             ! MODIFY BOTH OUTPUT FILE AND VARIABLES BELOW !                 #
###############################################################################

# TRAINING PARAMETERS

# ENV VARIABLES


# LOGGING SETUP

# export PYTHONPATH=${PYTHONPATH}:${PAPER_REPO}
# PYTHONPATH=${PAPER_REPO}

python3 explain.py --attacks_file data/attacks_val_no-letters.json
