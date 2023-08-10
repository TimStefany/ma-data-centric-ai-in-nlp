#!/bin/bash
#SBATCH --job-name model_experiment_bert
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=

hostname

echo "starting experiment\n\n"
python run_experiments_on_model.py -m bert-base-cased -lr 3e-5