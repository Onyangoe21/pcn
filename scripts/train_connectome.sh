#!/bin/bash

python -m tools.train_eval \
  --batch_size 32 \
  --complex_type path \
  --conv_type B \
  --dataset CONNECTOME \
  --device 0 \
  --drop_rate 0.4 \
  --emb_dim 16 \
  --epochs 100 \
  --eval_metric accuracy \
  --exp_name pcn-connectome \
  --final_readout sum \
  --graph_norm bn \
  --lr 0.001 \
  --lr_scheduler StepLR \
  --lr_scheduler_decay_rate 0.2 \
  --lr_scheduler_decay_steps 50 \
  --model sparse_cin \
  --nonlinearity relu \
  --num_layers 4 \
  --readout sum \
  --task_type classification \
  --train_eval_period 20
