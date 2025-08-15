# 文件路径: tasks/parity/dynamic_training/train_st_ctm.py
# (完整替换后的内容)

import argparse
import math
import multiprocessing
import os
import random
import torch
import numpy as np
import time  # 引入time模块用于调试

# 确保在导入pyplot之前设置后端
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torchvision
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# +++ 修改：从新的隔离文件中导入数据集 +++
from tasks.parity.dynamic_training.st_datasets import InMemoryParityDataset, TransitionDataset
# +++++++++++++++++++++++++++++++++++++

from tasks.parity.dynamic_training.st_ctm import ContinuousThoughtMachine as ST_CTM
from models.utils import reshape_predictions
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import parity_loss
from utils.schedulers import WarmupCosineAnnealingLR

from torch.utils.data import DataLoader

torchvision.disable_beta_transforms_warning()
torch.serialization.add_safe_globals([argparse.Namespace])


def prepare_st_ctm_model(prediction_reshaper, args, device):
    """
    一个专门用于初始化我们的ST_CTM的辅助函数。
    """
    model = ST_CTM(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=args.backbone_type,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=args.out_dims,
        prediction_reshaper=prediction_reshaper,
        dropout=args.dropout,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=0,
    ).to(device)
    return model


def generate_and_filter_data(ensemble_models, data_loader, args, device, prediction_reshaper):
    """
    [性能优化版 V2 - 批量传输 & 显存控制]
    阶段一和阶段二：生成和筛选优质思考片段。
    通过在GPU上累积数据并批量传输到CPU来减少通信开销。
    """
    all_filtered_data_cpu = []
    total_samples_generated = 0

    # 在GPU上临时累积数据的列表
    gpu_accumulator = []
    # 设置一个阈值，比如累积多少个批次的数据后再一次性传输
    ACCUMULATION_THRESHOLD = 4  # 累积16个批次的数据，可以根据您的显存大小调整

    pbar = tqdm(total=len(ensemble_models) * len(data_loader),
                desc="[Phase 1&2] Generating & Filtering Data")

    for model_idx, model in enumerate(ensemble_models):
        model.eval()
        original_iterations = model.iterations
        model.iterations = args.data_gen_ticks

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device,
                                                                               non_blocking=True)  # 使用 non_blocking=True
            batch_size = inputs.size(0)

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16,
                                    enabled=args.use_amp):
                    state_history_tensors = model(inputs, return_state_history=True)

                    if not state_history_tensors:
                        pbar.update(1)
                        continue

                # ... (此处数据筛选逻辑与之前版本完全相同) ...
                all_predictions = state_history_tensors["prediction"].float().permute(1, 2, 0)
                all_certainties = state_history_tensors["certainty"].float().permute(1, 2, 0)
                reshaped_preds_for_loss = all_predictions.reshape(batch_size, args.parity_sequence_length, 2,
                                                                  args.data_gen_ticks).permute(0, 3, 1, 2).reshape(-1,
                                                                                                                   args.parity_sequence_length,
                                                                                                                   2)
                expanded_targets = targets.unsqueeze(1).expand(-1, args.data_gen_ticks, -1).reshape(-1,
                                                                                                    args.parity_sequence_length)
                per_element_loss = torch.nn.functional.cross_entropy(
                    reshaped_preds_for_loss.reshape(-1, 2), expanded_targets.reshape(-1), reduction='none'
                ).reshape(batch_size, args.data_gen_ticks, args.parity_sequence_length)
                losses_all_ticks = per_element_loss.mean(dim=2)
                certainties_all_ticks = all_certainties[:, 1, :]
                losses_tm1, losses_t, losses_tp1 = losses_all_ticks[:, :-2], losses_all_ticks[:, 1:-1], \
                    losses_all_ticks[:, 2:]
                certs_tm1, certs_t = certainties_all_ticks[:, :-2], certainties_all_ticks[:, 1:-1]
                cond1 = (losses_t < losses_tm1) & (certs_t > certs_tm1)
                cond2 = (losses_tp1 < losses_t)
                good_indices_mask = cond1 & cond2
                batch_indices, tick_indices_t = torch.where(good_indices_mask)

                if batch_indices.numel() > 0:
                    num_found = batch_indices.numel()
                    total_samples_generated += num_found

                    input_tick_indices = tick_indices_t
                    target_tick_indices = tick_indices_t + 2

                    batch_data = {}
                    batch_data["input_kv"] = model.compute_features(inputs)[batch_indices].detach()
                    batch_data["targets"] = targets[batch_indices].detach()

                    for key, tensor_history in state_history_tensors.items():
                        if key.startswith('recursive_sync'): continue
                        batch_data[f"input_{key}"] = tensor_history[input_tick_indices, batch_indices].detach()
                        batch_data[f"target_{key}"] = tensor_history[target_tick_indices, batch_indices].detach()

                    # 将筛选出的数据累积在GPU列表中
                    gpu_accumulator.append(batch_data)

            # 检查是否达到了传输阈值
            if len(gpu_accumulator) >= ACCUMULATION_THRESHOLD:
                tqdm.write(f"Accumulation threshold reached. Transferring {len(gpu_accumulator)} batches to CPU...")
                # 批量合并和传输
                if gpu_accumulator:
                    consolidated_batch = {}
                    keys = gpu_accumulator[0].keys()
                    for key in keys:
                        # 合并张量并立即转移到CPU
                        consolidated_batch[key] = torch.cat([d[key] for d in gpu_accumulator], dim=0).cpu()
                    all_filtered_data_cpu.append(consolidated_batch)

                    # --- 核心修改 ---
                    # 1. 清空GPU累积器列表
                    gpu_accumulator.clear()

                    # 2. (可选但推荐) 显式删除中间变量，帮助Python垃圾回收
                    del consolidated_batch

                    # 3. (关键步骤) 强制PyTorch释放缓存的显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # ---------------

            pbar.update(1)

        model.iterations = original_iterations

    # 在所有循环结束后，处理剩余在累积器中的数据
    if gpu_accumulator:
        tqdm.write(f"Transferring remaining {len(gpu_accumulator)} batches to CPU...")
        consolidated_batch = {}
        keys = gpu_accumulator[0].keys()
        for key in keys:
            consolidated_batch[key] = torch.cat([d[key] for d in gpu_accumulator], dim=0).cpu()
        all_filtered_data_cpu.append(consolidated_batch)
        gpu_accumulator.clear()

    pbar.close()

    if not all_filtered_data_cpu:
        tqdm.write("[WARNING] No expert transition samples were generated in this epoch.")
        return None

    tqdm.write("Consolidating final data on CPU...")
    final_expert_data = {}
    keys = all_filtered_data_cpu[0].keys()
    for key in keys:
        final_expert_data[key] = torch.cat([d[key] for d in all_filtered_data_cpu], dim=0)

    tqdm.write(f"Generated and filtered a total of {final_expert_data['targets'].shape[0]} expert transition samples.")
    return final_expert_data


def define_state_loss(predicted_states, target_states):
    """
    定义用于比较两个状态包的损失函数。
    """
    total_loss = 0
    total_loss += torch.nn.functional.mse_loss(predicted_states[0]["activated_state"], target_states["activated_state"])
    total_loss += torch.nn.functional.mse_loss(predicted_states[1]["activated_state"], target_states["activated_state"])
    total_loss += torch.nn.functional.mse_loss(predicted_states[0]["attention_output"],
                                               target_states["attention_output"])
    total_loss += torch.nn.functional.mse_loss(predicted_states[1]["prediction"], target_states["prediction"])
    return total_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a State-Transition CTM on the Parity Task")
    # ... (参数部分与您文件中的保持一致，此处省略以保持简洁)
    # --- 原始模型与任务参数 ---
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm'],
                        help='Model type must be CTM for this paradigm.')
    parser.add_argument('--parity_sequence_length', type=int, default=64, help='Sequence length for parity task.')
    parser.add_argument('--d_model', type=int, default=1024, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=512, help='Dimension of the input projection.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear.')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons for output sync.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons for action sync.')
    parser.add_argument('--neuron_select_type', type=str, default='random',
                        help='Protocol for selecting neuron subset.')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of pre-activation history for NLMs.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep NLMs.')
    parser.add_argument('--memory_hidden_dims', type=int, default=16, help='Hidden dimensions for deep NLMs.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply normalization in NLMs.')
    parser.add_argument('--positional_embedding_type', type=str, default='custom-rotational-1d',
                        help='Type of positional embedding.')
    parser.add_argument('--backbone_type', type=str, default='parity_backbone',
                        help='Type of backbone feature extractor.')

    # --- 新范式的特定参数 ---
    parser.add_argument('--num_ensemble_models', type=int, default=10,
                        help='Number of models in the ensemble for diverse data generation.')
    parser.add_argument('--data_gen_ticks', type=int, default=200,
                        help='Number of ticks for the long forward pass during data generation.')
    parser.add_argument('--lambda_expert_ratio', type=float, default=0.5,
                        help='Ratio of expert data vs. self-play data in a training batch.')

    # --- 训练与通用参数 ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--training_iterations', type=int, default=200001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='Type of learning rate scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')

    # --- 后勤参数 ---
    parser.add_argument('--log_dir', type=str, default='logs/parity_dynamic', help='Directory for logging.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--save_every', type=int, default=10000, help='Save checkpoint frequency.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True, help='AMP autocast.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    zip_python_code(os.path.join(args.log_dir, 'repo_state.zip'))
    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
        print(args, file=f)

    if args.device[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    print(f"Running new CTM training paradigm on device: {device}")

    writer = SummaryWriter(log_dir=f'{args.log_dir}/tensorboard')

    args.iterations = 2
    prediction_reshaper = [args.parity_sequence_length, 2]
    args.out_dims = args.parity_sequence_length * 2

    model = prepare_st_ctm_model(prediction_reshaper, args, device)

    try:
        pseudo_inputs = torch.randn(1, args.parity_sequence_length).to(device)
        model(pseudo_inputs)
        print(f'Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    except Exception as e:
        print(f"Could not perform pseudo-forward pass: {e}")

    print("Initializing InMemoryParityDataset...")
    train_data = InMemoryParityDataset(sequence_length=args.parity_sequence_length, length=100)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              persistent_workers=True if args.num_workers > 0 else False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations)

    print("\n" + "=" * 50)
    print("      New CTM Training Paradigm - Starting...")
    print("=" * 50 + "\n")

    print("Initializing ensemble models for data generation...")
    ensemble_models = [prepare_st_ctm_model(prediction_reshaper, args, device) for i in range(args.num_ensemble_models)]
    for i, m in enumerate(ensemble_models):
        set_seed(args.seed + i + 1)
        m.eval()

    set_seed(args.seed)

    expert_transitions = generate_and_filter_data(ensemble_models, train_loader, args, device, prediction_reshaper)

    if expert_transitions is None:
        print("Stopping training as no expert data was generated.")
        exit()

    expert_dataset = TransitionDataset(expert_transitions)
    expert_loader = DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    expert_iterator = iter(expert_loader)

    with tqdm(total=args.training_iterations, initial=0, desc="[Phase 3] Training ST-CTM Unit") as pbar:
        for bi in range(args.training_iterations):
            try:
                batch_data = next(expert_iterator)
            except StopIteration:
                expert_iterator = iter(expert_loader)
                batch_data = next(expert_iterator)

            # --- Move data to device ---
            # 这部分需要根据 TransitionDataset 的 __getitem__ 返回结构来调整
            # 假设返回的是 (input_dict, target_dict)
            input_dict = {k.replace('input_', ''): v.to(device) for k, v in batch_data.items() if
                          k.startswith('input_')}
            target_dict = {k.replace('target_', ''): v.to(device) for k, v in batch_data.items() if
                           k.startswith('target_')}

            # --- 执行双步前向传播 ---
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16,
                                enabled=args.use_amp):
                # model.run_two_ticks 是一个需要你在 st_ctm.py 中实现的新方法
                predicted_final_state, _ = model.run_two_ticks(input_dict)
                loss = define_state_loss(predicted_final_state, target_dict)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            writer.add_scalar('Loss/state_transition', loss.item(), bi)
            writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], bi)

            pbar.set_description(
                f"[Phase 3] Training ST-CTM Unit | Iter {bi + 1}/{args.training_iterations} | Loss: {loss.item():.4f}")
            pbar.update(1)

            if (bi + 1) % args.save_every == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iteration': bi,
                }, os.path.join(args.log_dir, f'checkpoint_{bi + 1}.pt'))

    print("Training of ST-CTM Unit complete.")
    writer.close()
