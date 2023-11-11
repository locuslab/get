torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_endpoint=localhost:$2 \
    train.py \
    --epochs 1000              \
    --global_batch_size 128    \
    --grad 6                   \
    --sup_gap 1                \
    --f_max_iter 0             \
    --eval_f_max_iter 6        \
    --norm_type none           \
    --data_path YOUR_DATA_PATH \
    --stat_path YOUR_STAT_PATH \
    ${@:3}
