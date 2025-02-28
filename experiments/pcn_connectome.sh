#!/bin/bash

python -m experiments.train_eval_connectome \
  --batch_size 32 \
  --complex_type path \
  --dataset CONNECTOME \
  --device 0 \
  --dropout 0.4 \
  --emb_dim 16 \
  --epochs 100 \
  --eval_metric accuracy \
  --exp_name pcn-connectome \
  --lr 0.001 \
  --model sparse_cin \
  --num_layers 4
