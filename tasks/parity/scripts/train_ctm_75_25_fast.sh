#!/bin/bash

# --- 配置区 ---
TRAINING_STEPS=200001
T_ITERATIONS=75
MEMORY_LENGTH=25
BATCH_SIZE=512
NUM_WORKERS=64

# 关键修改：将学习率进一步降低，以确保训练后期的稳定性
LEARNING_RATE=0.0004

# 日志和随机种子设置
RUN=1
LOG_DIR="logs/parity/run${RUN}/ctm_${T_ITERATIONS}_${MEMORY_LENGTH}_fast"
SEED=$((RUN - 1))

# --- 执行区 ---
python -m tasks.parity.train \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --iterations $T_ITERATIONS \
    --memory_length $MEMORY_LENGTH \
    --parity_sequence_length 64  \
    --n_test_batches 20 \
    --d_model 1024 \
    --d_input 512 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --heads 8 \
    --memory_hidden_dims 16 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --positional_embedding_type="custom-rotational-1d" \
    --backbone_type="parity_backbone" \
    --no-full_eval \
    --weight_decay 0.0 \
    --gradient_clipping 0.9 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --dataset "parity" \
    --batch_size $BATCH_SIZE \
    --batch_size_test 256 \
    --lr=$LEARNING_RATE \
    --training_iterations $TRAINING_STEPS \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 10000 \
    --no-reload \
    --no-reload_model_only \
    --device 0 \
    --use_amp \
    --num_workers $NUM_WORKERS \
    --neuron_select_type "random"
