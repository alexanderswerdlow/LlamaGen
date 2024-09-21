#!/bin/bash

NUM_GPUS=6
NUM_NODES=1
torchrun \
--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
tokenizer/tokenizer_image/vq_train.py \
--vq-model="VQ-16" \
--dataset imagenet \
--codebook-size 1024 \
--codebook-embed-dim 8 \
--image-size 256 \
--global-batch-size $((8 * NUM_GPUS * NUM_NODES)) \
--gradient-accumulation-steps 1 \
--ckpt-every 5000 \
--data-path '/home/mprabhud/phd_projects/mdlm/clevr_imgs/' \
--cloud-save-path '/scratch/aswerdlo/llamagen/outputs' \
--disc-start 10000 \
--no-local-save