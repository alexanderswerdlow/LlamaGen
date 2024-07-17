#!/bin/bash
#SBATCH --job-name=llama_vq8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --constraint=A6000|A5000
#SBATCH --comment=aswerdlo
#SBATCH --time=48:00:00
#SBATCH --partition=deepaklong
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.out

# sbatch scripts/tokenizer/train_vq_v4.sh
export nnodes=$SLURM_NNODES
export nproc_per_node=$SLURM_GPUS_PER_NODE

echo "ibstatus: $(ibstatus)"
echo "ibdev2netdev: $(ibdev2netdev)"
echo "rdma device: $(rdma link)"
export LOGLEVEL=INFO
# choose one node as the master node for ddp training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# random choose a port between 30000:50000 for master node communitication
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo WORLD_SIZE: $WORLD_SIZE
echo RANK: $RANK
echo LOCAL_RANK: $LOCAL_RANK

echo "environment: $(env | grep NCCL)"

srun --label torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
tokenizer/tokenizer_image/vq_train.py \
--vq-model="VQ-8" \
--dataset face \
--codebook-size 1024 \
--codebook-embed-dim 256 \
--image-size 256 \
--global-batch-size $((8 * nproc_per_node * nnodes)) \
--gradient-accumulation-steps 1 \
--ckpt-every 2000 \
--cloud-save-path '/grogu/user/mprabhud/aswerdlo/llamagen/outputs' \
--disc-start 6000 \
--no_local_save