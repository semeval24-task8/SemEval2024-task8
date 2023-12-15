#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=16000

module purge
module load scikit-learn

pip install stanza stanza-batch scikit-learn pandas tqdm

python3 --version

python3 linguistic-batch.py -i ~/data/SubtaskA/subtaskA_train_monolingual.jsonl -o ling_a_train_mono.csv -bs 10

deactivate