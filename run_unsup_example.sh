#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --stronger_train_file data/wiki1m_for_simcse.txt \
    --output_dir result/with_swav \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model sickr_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --pooler_q_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --bank_size 3200 \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_clustering \
    --mlm_weight 0.1 \
    --nmb_prototypes 3000 \
    --epsilon 0.05 \
    --cutoff_rate 0.15 \
    --fp16 \
    "$@"
