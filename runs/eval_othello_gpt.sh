#!/bin/bash -l
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -l h_rt=4:59:59
#$ -o logs/$JOB_ID_eval_othello_gpt.log
#$ -j y
#$ -m e
#$ -M sschusle@bu.edu

module load miniconda
module load cuda/12.2

conda activate othello

echo "Evaluating checkpoint ${1} with causal_mask_limit=${2}"

python evaluate_gpt_othello.py --checkpoint_path ${1} --causal_mask_limit ${2}

