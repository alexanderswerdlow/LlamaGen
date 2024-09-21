#!/bin/bash

NUM_GPUS=${NUM_GPUS:-6}
NUM_NODES=${NUM_NODES:-1}
torchrun \
--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
tokenizer/tokenizer_image/vq_train.py \
--finetune \
--disc-start 0 \
--vq-ckpt /home/mprabhud/aswerdlo/repos/mdlm/ckpts/vq_ds8_c2i.pt \
--vq-model="VQ-8" \
--dataset imagenet \
--image-size 128 \
--global-batch-size $((8 * NUM_GPUS * NUM_NODES)) \
--gradient-accumulation-steps 1 \
--ckpt-every 2500 \
--data-path '/home/mprabhud/phd_projects/mdlm/clevr_imgs/' \
--cloud-save-path '/scratch/aswerdlo/llamagen/outputs' \
--disc-start 10000 \
--no-local-save