#!/bin/bash -l
#$ -P compgen
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=4:00:00
#$ -o logs/$JOB_ID_train_probe_layer7_limited_attn.log
#$ -j y
#$ -m e
#$ -M sschusle@bu.edu

module load miniconda
module load cuda/12.2

conda activate othello

for Y in {4,5,6,7,8}
do
echo "LAYER: ${Y}"
python train_probe_othello.py \
    --layer ${Y} \
    --epo 16 \
    --mid_dim 512 \
    --causal_mask_limit 4 \
    --checkpoint_path ckpts/gpt_limited_attention_4.ckpt \
    --twolayer \
    --exp_name limited_attn_layer${Y}

done
