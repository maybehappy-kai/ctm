#!/bin/bash

# --- 配置区 ---
# 实验标识
RUN=1
LOG_DIR="logs/parity_dynamic/run${RUN}"
SEED=$((RUN - 1))

# 模型核心参数 (与原论文75-tick, 25-mem CTM对齐以便比较)
D_MODEL=1024
D_INPUT=512
HEADS=8
SYNAPSE_DEPTH=1
MEMORY_LENGTH=25
N_SYNCH=32
MEMORY_HIDDEN_DIMS=16

# 新范式特定参数
NUM_ENSEMBLE_MODELS=16   # 用于生成多样化专家数据的模型数量
DATA_GEN_TICKS=250     # 数据生成时的长前向传播步数
LAMBDA=0.5             # 专家数据与自迭代数据混合比例

# 训练参数
TRAINING_ITERATIONS=200001
BATCH_SIZE=128
LR=0.0001
WARMUP_STEPS=500
SAVE_EVERY=10000


# --- 执行区 ---
# 注意模块路径的变更
python -m tasks.parity.dynamic_training.train_st_ctm \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --model_type "ctm" \
    --d_model $D_MODEL \
    --d_input $D_INPUT \
    --heads $HEADS \
    --synapse_depth $SYNAPSE_DEPTH \
    --memory_length $MEMORY_LENGTH \
    --n_synch_out $N_SYNCH \
    --n_synch_action $N_SYNCH \
    --memory_hidden_dims $MEMORY_HIDDEN_DIMS \
    --num_ensemble_models $NUM_ENSEMBLE_MODELS \
    --data_gen_ticks $DATA_GEN_TICKS \
    --lambda_expert_ratio $LAMBDA \
    --training_iterations $TRAINING_ITERATIONS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --save_every $SAVE_EVERY \
    --device 1 \
    --use_amp \
    --num_workers 16
