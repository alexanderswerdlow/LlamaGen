#!/bin/bash

export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

NUM_GPUS=8
torchrun \
--nnodes=1 --nproc_per_node=$NUM_GPUS \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
tokenizer/tokenizer_image/vq_train.py \
--vq-model="VQ-16" \
--dataset imagenet \
--codebook-size 1024 \
--codebook-embed-dim 256 \
--image-size 256 \
--global-batch-size $((8 * NUM_GPUS)) \
--gradient-accumulation-steps 4 \
--ckpt-every 5000 \
--cloud-save-path '/grogu/user/mprabhud/aswerdlo/llamagen/outputs' \
--disc-start 5000 \
--data-path /grogu/datasets/imagenet/train