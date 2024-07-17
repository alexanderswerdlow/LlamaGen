# !/bin/bash
set -x

# bash scripts/tokenizer/train_vq_v2.sh

export nnodes=1
export nproc_per_node=8
export node_rank=0
export master_addr=127.0.0.1
export master_port=$RANDOM

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py \
--vq-model="VQ-8" \
--dataset imagenet \
--codebook-size 1024 \
--codebook-embed-dim 256 \
--image-size 256 \
--global-batch-size $((8 * nproc_per_node * nnodes)) \
--gradient-accumulation-steps 2 \
--ckpt-every 2000 \
--data-path '/compute/grogu-1-40/aswerdlo/data/vggface2/data/train,/grogu/user/mprabhud/data/diffusion/ffhq,/grogu/user/mprabhud/data/diffusion/sfhq' \
--cloud-save-path /grogu/user/mprabhud/aswerdlo/llamagen/outputs \
"$@"