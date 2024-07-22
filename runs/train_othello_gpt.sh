#!/bin/bash -l
#$ -l gpus=2
#$ -l gpu_memory=40G
#$ -l gpu_c=7.0
#$ -l h_rt=47:59:59
#$ -o logs/$JOB_ID_train_othello_gpt.log
#$ -j y
#$ -m e
#$ -M sschusle@bu.edu

module load miniconda
module load cuda/12.2

conda activate othello

python train_gpt_othello.py

