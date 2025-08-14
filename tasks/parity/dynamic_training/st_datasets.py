# 文件路径: tasks/parity/dynamic_training/st_datasets.py

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class InMemoryParityDataset(Dataset):
    """
    一个将所有数据预先生成并存储在内存中的ParityDataset版本，
    以消除数据加载瓶颈。
    """

    def __init__(self, sequence_length=64, length=100000):
        self.sequence_length = sequence_length
        self.length = length

        print(f"Pre-generating {length} samples for InMemoryParityDataset...")

        # 一次性生成所有数据和标签
        all_vectors = 2 * torch.randint(0, 2, (length, sequence_length)) - 1
        self.vectors = all_vectors.float()

        negatives = (self.vectors == -1).to(torch.long)
        cumsum = torch.cumsum(negatives, dim=1)
        self.targets = (cumsum % 2 != 0).to(torch.long)

        print("InMemoryParityDataset pre-generation complete.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # __getitem__ 现在只做快速的索引操作
        return self.vectors[idx], self.targets[idx]


class TransitionDataset(Dataset):
    """
    [重构优化版]
    存储批处理后的张量字典，并按索引返回单个样本的所有状态信息。
    """

    def __init__(self, expert_data_dict):
        # expert_data_dict 是 generate_and_filter_data 返回的张量字典
        self.data = expert_data_dict
        # 假设所有张量的第一个维度都是样本数
        self.num_samples = self.data['targets'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回一个字典，其中每个值都是对应索引处的张量切片
        sample = {key: tensor[idx] for key, tensor in self.data.items()}
        return sample
