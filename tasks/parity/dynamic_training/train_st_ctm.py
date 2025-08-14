import argparse
import math
import multiprocessing
import os
import random
import torch
import numpy as np

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

from data.custom_datasets import ParityDataset
from tasks.parity.utils import prepare_model
from utils.housekeeping import set_seed, zip_python_code
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
from torch.utils.tensorboard import SummaryWriter

# from tasks.parity.utils import prepare_model
from tasks.parity.dynamic_training.st_ctm import ContinuousThoughtMachine as ST_CTM
from models.utils import reshape_predictions
from utils.losses import parity_loss

from torch.utils.data import Dataset, DataLoader

torchvision.disable_beta_transforms_warning()
# 允许安全地加载包含argparse.Namespace的checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])


def prepare_st_ctm_model(prediction_reshaper, args, device):
    """
    一个专门用于初始化我们的ST_CTM的辅助函数。
    """
    model = ST_CTM(
        # ... (将 prepare_model 中的所有参数复制到这里) ...
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
        n_random_pairing_self=0,  # 在parity任务中未使用
    ).to(device)
    return model


def calculate_loss_and_certainty(state_package, targets, prediction_reshaper):
    """
    一个辅助函数，用于从一个状态包中计算损失和确定性。
    """
    # 为单个样本添加批次维度 (B=1)，以匹配工具函数期望的输入形状
    predictions = state_package["prediction"].unsqueeze(0).unsqueeze(-1)
    certainties = state_package["certainty"].unsqueeze(0).unsqueeze(-1)

    predictions = reshape_predictions(predictions, prediction_reshaper)

    # Parity loss for a single tick. Note: use_most_certain is False as we provide only one tick.
    loss, _ = parity_loss(predictions, certainties, targets, use_most_certain=False)

    # 确定性是 state_package 中 certainty 张量的第二个元素
    certainty = state_package["certainty"][1].item()

    return loss.item(), certainty


# """
# 修改
# """
def generate_and_filter_data(ensemble_models, data_loader, args, device, prediction_reshaper):
    """
    阶段一和阶段二：生成和筛选优质思考片段。
    采用“双步思考单元”的逻辑进行筛选。
    [最终优化版：全批次向量化索引与提取]
    """
    expert_transitions = []
    pbar = tqdm(total=len(ensemble_models) * len(data_loader), desc="[Phase 1&2] Generating & Filtering Data")

    for model in ensemble_models:
        model.eval()
        original_iterations = model.iterations
        model.iterations = args.data_gen_ticks

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            with torch.no_grad():
                # 1. 一次性为整个批次生成状态历史 (这是一个 list of dicts)
                state_history = model(inputs, return_state_history=True)
                if not state_history:
                    pbar.update(1)
                    continue

                # 2. 向量化计算所有ticks的损失和确定性 (与之前相同)
                all_predictions = torch.stack([s['prediction'] for s in state_history], dim=2)
                all_certainties = torch.stack([s['certainty'] for s in state_history], dim=2)

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

                # 3. 向量化筛选“优质”转换 (与之前相同)
                losses_tm1, losses_t, losses_tp1 = losses_all_ticks[:, :-2], losses_all_ticks[:, 1:-1], \
                losses_all_ticks[:, 2:]
                certs_tm1, certs_t = certainties_all_ticks[:, :-2], certainties_all_ticks[:, 1:-1]
                cond1 = (losses_t < losses_tm1) & (certs_t > certs_tm1)
                cond2 = (losses_tp1 < losses_t)
                good_indices_mask = cond1 & cond2
                batch_indices, tick_indices_t = torch.where(good_indices_mask)

                # 4. 彻底的批次化提取
                if batch_indices.numel() > 0:
                    input_tick_indices = tick_indices_t
                    target_tick_indices = tick_indices_t + 2

                    # --- 核心性能优化：将历史列表转换为历史字典 ---
                    history_dict = {k: torch.stack(
                        [s[k] if not isinstance(s[k], tuple) else torch.stack(s[k], dim=0) for s in state_history],
                        dim=0) for k in state_history[0].keys()}
                    # 结果是，例如 history_dict['activated_state'] 的形状为 (T, B, D)

                    # --- 使用高级索引一次性提取所有数据 ---
                    kv_context = model.compute_features(inputs)[batch_indices].detach().cpu()
                    targets_context = targets[batch_indices].detach().cpu()

                    input_states_batch = {}
                    target_states_batch = {}
                    for key, tensor_history in history_dict.items():
                        if key.startswith('recursive_sync'):
                            # 对元组的特殊处理
                            alpha_in = tensor_history[input_tick_indices, 0, batch_indices].detach().cpu()
                            beta_in = tensor_history[input_tick_indices, 1, batch_indices].detach().cpu()
                            input_states_batch[key] = (alpha_in, beta_in)

                            alpha_tgt = tensor_history[target_tick_indices, 0, batch_indices].detach().cpu()
                            beta_tgt = tensor_history[target_tick_indices, 1, batch_indices].detach().cpu()
                            target_states_batch[key] = (alpha_tgt, beta_tgt)
                        else:
                            input_states_batch[key] = tensor_history[input_tick_indices, batch_indices].detach().cpu()
                            target_states_batch[key] = tensor_history[target_tick_indices, batch_indices].detach().cpu()

                    # --- 在CPU上快速重组为最终列表 ---
                    for i in range(batch_indices.numel()):
                        input_state = {k: v[i] if not isinstance(v, tuple) else (v[0][i], v[1][i]) for k, v in
                                       input_states_batch.items()}
                        target_state = {k: v[i] if not isinstance(v, tuple) else (v[0][i], v[1][i]) for k, v in
                                        target_states_batch.items()}
                        context = {"kv": kv_context[i], "targets": targets_context[i]}
                        expert_transitions.append((input_state, target_state, context))

            pbar.update(1)
        model.iterations = original_iterations

    pbar.close()
    print(f"Generated and filtered {len(expert_transitions)} expert transition samples.")
    return expert_transitions

# """
# 修改
# """
def collate_fn(batch):
    """
    自定义的collate_fn，用于将状态包列表正确地堆叠成批次。
    """
    input_states, target_states, contexts = zip(*batch)

    collated_input = {}
    collated_target = {}
    collated_context = {}

    # 提取第一个样本以获取所有的键
    input_keys = input_states[0].keys()
    target_keys = target_states[0].keys()
    context_keys = contexts[0].keys()

    for key in input_keys:
        # 特殊处理元组形式的递归同步状态
        if key.startswith('recursive_sync'):
            alpha_list = [state[key][0] for state in input_states]
            beta_list = [state[key][1] for state in input_states]
            collated_input[key] = (torch.stack(alpha_list, dim=0), torch.stack(beta_list, dim=0))
        else:
            collated_input[key] = torch.stack([state[key] for state in input_states], dim=0)

    for key in target_keys:
        if key.startswith('recursive_sync'):
            alpha_list = [state[key][0] for state in target_states]
            beta_list = [state[key][1] for state in target_states]
            collated_target[key] = (torch.stack(alpha_list, dim=0), torch.stack(beta_list, dim=0))
        else:
            collated_target[key] = torch.stack([state[key] for state in target_states], dim=0)

    for key in context_keys:
        collated_context[key] = torch.stack([ctx[key] for ctx in contexts], dim=0)

    return collated_input, collated_target, collated_context


# """
# 修改
# """
def define_state_loss(predicted_states, target_states):
    """
    定义用于比较两个状态包的损失函数。
    """
    total_loss = 0

    # 1. 后激活值 (z) 的损失
    total_loss += torch.nn.functional.mse_loss(predicted_states[0]["activated_state"], target_states["activated_state"])
    total_loss += torch.nn.functional.mse_loss(predicted_states[1]["activated_state"], target_states["activated_state"])

    # 2. 注意力输出 (o) 的损失
    total_loss += torch.nn.functional.mse_loss(predicted_states[0]["attention_output"],
                                               target_states["attention_output"])

    # 3. 预测 (y) 的损失
    total_loss += torch.nn.functional.mse_loss(predicted_states[1]["prediction"], target_states["prediction"])

    # 还可以添加其他状态的损失，但以上是最关键的
    return total_loss


# """
# # 修改
# """
class TransitionDataset(Dataset):
    """
    一个用于存储和加载 (输入状态, 目标状态) 对的自定义数据集。
    """

    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        input_state, target_state, context = self.transitions[idx]
        return input_state, target_state, context

def parse_args():
    # 复用原始的参数解析，并为新范式添加特定参数
    parser = argparse.ArgumentParser(description="Train a State-Transition CTM on the Parity Task")

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
    # --- 1. 初始化和设置 ---
    args = parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # 保存代码和参数以保证可复现性
    zip_python_code(os.path.join(args.log_dir, 'repo_state.zip'))
    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
        print(args, file=f)

    # 设置设备
    if args.device[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    print(f"Running new CTM training paradigm on device: {device}")

    # --- 2. 准备模型 (ST-CTM Unit) ---
    # 注意：这里的iterations参数将被用于推理，而不是固定的训练循环
    # 我们将使用“双步思考”单元，所以模型本身仍然定义为标准的CTM
    args.iterations = 2  # 设定为2以匹配“双步思考”单元
    prediction_reshaper = [args.parity_sequence_length, 2]
    args.out_dims = args.parity_sequence_length * 2

    # 初始化模型
    model = prepare_st_ctm_model(prediction_reshaper, args, device)

    # 打印模型参数量
    try:
        pseudo_inputs = torch.randn(1, args.parity_sequence_length).to(device)
        model(pseudo_inputs)
        print(f'Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    except Exception as e:
        print(f"Could not perform pseudo-forward pass: {e}")

    # --- 3. 准备数据加载器 ---
    # 我们仍然需要一个数据加载器来获取原始的奇偶校验问题实例
    train_data = ParityDataset(sequence_length=args.parity_sequence_length, length=100000)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)

    # --- 4. 准备优化器和调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations)

    # --- 5. 新的训练范式实现区域 ---
    print("\n" + "=" * 50)
    print("      New CTM Training Paradigm - Starting...")
    print("=" * 50 + "\n")

    # ------------------ 阶段一 & 二: 数据生成与筛选 ------------------
    # 创建一个模型集成用于生成多样化数据
    ensemble_models = []
    for i in range(args.num_ensemble_models):
        print(f"Initializing ensemble model {i + 1}/{args.num_ensemble_models}...")
        # 注意：每个模型都需要自己的随机种子以保证多样性
        set_seed(args.seed + i)
        ensemble_model = prepare_st_ctm_model(prediction_reshaper, args, device)
        # 这里可以加载不同的预训练权重，但目前我们从随机初始化开始
        ensemble_model.eval()  # 设置为评估模式
        ensemble_models.append(ensemble_model)

    # 重置主模型的种子
    set_seed(args.seed)

    expert_transitions = generate_and_filter_data(ensemble_models, train_loader, args, device, prediction_reshaper)

    # 将筛选出的片段封装成数据集和数据加载器
    expert_dataset = TransitionDataset(expert_transitions)
    # 注意：这里的 batch_size 可以设置得很大，因为训练是并行的
    expert_loader = DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)


    # ------------------ 阶段三: 并行化监督训练 ------------------
    model.train()  # 确保主模型处于训练模式
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    expert_iterator = iter(expert_loader)

    with tqdm(total=args.training_iterations, initial=0, desc="[Phase 3] Training ST-CTM Unit") as pbar:
        for bi in range(args.training_iterations):
            try:
                input_states, target_states, contexts = next(expert_iterator)
            except StopIteration:
                expert_iterator = iter(expert_loader)
                input_states, target_states, contexts = next(expert_iterator)

            # 将所有数据移动到设备
            input_states = {k: v.to(device) if isinstance(v, torch.Tensor) else (v[0].to(device), v[1].to(device)) for
                            k, v in input_states.items()}
            target_states = {k: v.to(device) for k, v in target_states.items()}
            contexts = {k: v.to(device) for k, v in contexts.items()}

            # --- 执行双步前向传播 ---
            # 这是一个简化的前向传播，我们需要构建一个能执行双步的函数
            # 这里我们直接在循环中实现

            # 注入静态KV
            model.kv_features = contexts["kv"].squeeze(1)

            # --- Tick 1 ---
            # 从输入状态包中提取初始状态
            state_trace_t0 = input_states["state_trace"]
            activated_state_t0 = input_states["activated_state"]
            attn_out_t0 = input_states["attention_output"]
            decay_alpha_action_t0, decay_beta_action_t0 = input_states["recursive_sync_action"]
            decay_alpha_out_t0, decay_beta_out_t0 = input_states["recursive_sync_out"]

            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16,
                                enabled=args.use_amp):
                # 手动执行第一个tick的计算
                pre_synapse_input_t1 = torch.cat((attn_out_t0, activated_state_t0), dim=-1)
                state_t1 = model.synapses(pre_synapse_input_t1)
                state_trace_t1 = torch.cat((state_trace_t0[:, :, 1:], state_t1.unsqueeze(-1)), dim=-1)
                activated_state_t1 = model.trace_processor(state_trace_t1)

                # ... (此处省略了第一个tick的同步和注意力计算，因为我们只需要最终输出)

                # --- Tick 2 ---
                # 为了简化，我们直接使用 target_states 中的 attention_output 作为 o_t1
                # 这是一个近似，一个更完整的实现会重新计算它
                pre_synapse_input_t2 = torch.cat((target_states["attention_output"], activated_state_t1), dim=-1)
                state_t2 = model.synapses(pre_synapse_input_t2)
                state_trace_t2 = torch.cat((state_trace_t1[:, :, 1:], state_t2.unsqueeze(-1)), dim=-1)
                activated_state_t2 = model.trace_processor(state_trace_t2)

                # 重新计算第二个tick的同步和预测
                r_out = torch.exp(-torch.clamp(model.decay_params_out, 0, 15)).unsqueeze(0).expand(args.batch_size, -1)
                _, d_alpha_out_t1, d_beta_out_t1 = model.compute_synchronisation(activated_state_t1, decay_alpha_out_t0,
                                                                                 decay_beta_out_t0, r_out, 'out')
                sync_out_t2, _, _ = model.compute_synchronisation(activated_state_t2, d_alpha_out_t1, d_beta_out_t1,
                                                                  r_out, 'out')
                prediction_t2 = model.output_projector(sync_out_t2)

                # --- 定义损失 ---
                # 简化的损失，只监督最终的预测
                loss = torch.nn.functional.mse_loss(prediction_t2, target_states["prediction"])

            # --- 反向传播 ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_description(
                f"[Phase 3] Training ST-CTM Unit | Iter {bi + 1}/{args.training_iterations} | Loss: {loss.item():.4f}")
            pbar.update(1)

            # --- 定期保存模型 ---
            if (bi + 1) % args.save_every == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'iteration': bi,
                }, os.path.join(args.log_dir, f'checkpoint_{bi + 1}.pt'))

    print("Training of ST-CTM Unit complete.")
