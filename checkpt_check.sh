#!/bin/bash


source ~/CoTFormer/iridis/env.sh


export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "--- Starting Integrity Check for Step 500 ---"



/scratch/ab3u21/cotformer-env/bin/python eval.py \
  --checkpoint /scratch/ab3u21/exps/owt2/cotformer_full_depth/Deadlock_RAM_test_cotformer_full_depth_27_layer_test_bs8x16_seqlen256/ckpt_300.pt \
  --config_format base \
  --dataset owt2 \
  --data_dir "/scratch/ab3u21/datasets" \
  --model cotformer_full_depth \
  --n_layer 27 \
  --n_repeat 5 \
  --distributed_backend None \
  --device cuda:0 \
  --eval_sample_size 1024
