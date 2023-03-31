#!/bin/bash
#SBATCH --job-name=test_val_no-letters_lime
#SBATCH --output=outputs/test_val_no-letters_lime.txt
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

python3 attack_dataset.py \
    --split "test" \
    --substitution_file data/substitutions_val_no-letters_lime.json
