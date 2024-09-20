#!/bin/bash

NUM_GPUS=8
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
--global-batch-size $((16 * NUM_GPUS * NUM_NODES)) \
--gradient-accumulation-steps 1 \
--ckpt-every 5000 \
--data-path '/compute/grogu-1-40/aswerdlo/data/vggface2/data/train,/grogu/user/mprabhud/data/diffusion/ffhq,/grogu/user/mprabhud/data/diffusion/celeba_hq' \
--cloud-save-path '/grogu/user/mprabhud/aswerdlo/llamagen/outputs' \
--disc-start 10000 \
--no-local-save