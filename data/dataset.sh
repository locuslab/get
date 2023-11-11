torchrun --standalone --nproc_per_node=4 generate.py \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
    --outdir=YOUR_UNCOND_DATA_PATH \
    --seeds=0-999999 \
    --batch 250

torchrun --standalone --nproc_per_node=4 cond_generate.py \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --outdir=YOUR_COND_DATA_PATH   \
    --seeds=0-999999 \
    --batch 250

