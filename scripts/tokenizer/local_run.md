torchrun \
--nnodes=1 --nproc_per_node=1 \
tokenizer/tokenizer_image/vq_train.py \
--vq-model="VQ-8" \
--dataset face \
--data-path '/compute/grogu-1-40/aswerdlo/data/vggface2/data/train,/grogu/user/mprabhud/data/diffusion/ffhq,/grogu/user/mprabhud/data/diffusion/celeba_hq' \
--codebook-size 8192 \
--codebook-embed-dim 32 \
--image-size 512 \
--global-batch-size 4 \
--gradient-accumulation-steps 2 \
--ckpt-every 2000 \
--cloud-save-path '/grogu/user/mprabhud/aswerdlo/llamagen/debug' \
--disc-start 6000 \
--ema