torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_endpoint=localhost:$2 \
    eval.py                    \
    --eval_f_max_iter 6        \
    --norm_type none           \
    --stat_path YOUR_STAT_PATH \
    ${@:3}
