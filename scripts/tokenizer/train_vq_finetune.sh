# !/bin/bash
set -x

export nnodes=1
export nproc_per_node=6
export node_rank=0
export master_addr=127.0.0.1
export master_port=$RANDOM

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py \
--finetune \
--disc-start 0 \
--vq-ckpt /home/mprabhud/aswerdlo/repos/mdlm/ckpts/vq_ds8_c2i.pt \
--vq-model="VQ-8" \
--dataset imagenet \
--data-path /compute/grogu-1-40/aswerdlo/data/vggface2/data/train \
--cloud-save-path /home/mprabhud/aswerdlo/repos/lib/LlamaGen/outputs \
"$@"