#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=16000

module purge
module load scikit-learn

pip install stanza scikit-learn torch pandas tqdm

python3 --version

python3 information-redundancy.py -i ~/data/SubtaskB/subtaskB_train.jsonl -o info_red_b_train.csv

deactivate