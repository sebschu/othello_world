#!/bin/bash -l
#$ -P compgen
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=3:30:00
#$ -o logs/$JOB_ID_train_probe_layer7_replication.log
#$ -j y
#$ -m e
#$ -M sschusle@bu.edu

module load miniconda
module load cuda/12.2

conda activate othello

for Y in {2,4,8,16,32,64,128,256,512}
do
echo "MID DIM: ${Y}"
python train_probe_othello.py \
    --layer 7 \
    --epo 16 \
    --causal_mask_limit -1 \
    --checkpoint_path ckpts/gpt_synthetic.ckpt \
    --twolayer \
    --mid_dim $Y \
    --exp_name replication  

done
