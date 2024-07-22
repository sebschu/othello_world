import os
import math
import argparse
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--causal_mask_limit", default=-1, type=int)

    args = parser.parse_args()

    synthetic_or_championship = True  # True for training on the synthetic dataset

    othello = get_othello(ood_num=-1, data_root=None if 
                          synthetic_or_championship 
                          else "data/othello_championship", wthor=True)
    train_dataset = CharDataset(othello)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512, causal_mask_limit=args.causal_mask_limit)
    model = GPT(mconf)



    max_epochs = 250
    # initialize a trainer instance and kick off training
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=max_epochs, 
        batch_size=512*8,  # assuming 2 40G GPUs
        learning_rate=5e-4,
        lr_decay=True, 
        warmup_tokens=len(train_dataset)*train_dataset.block_size*5, 
        final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
        num_workers=0, 
        ckpt_path=f"./ckpts/gpt_at{t_start}.ckpt", 
    )
    trainer = Trainer(model, train_dataset, None, tconf)
    device = trainer.device
    print(t_start)
    trainer.train()



if __name__ == "__main__":
    main()