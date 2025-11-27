# data_loader.py

import torch
import numpy as np
import pickle
import h5py
import logging
from typing import Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader
from config import CONFIG


def load_and_preprocess_data(path: str, train_fraction: float = CONFIG["train_fraction"]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, int]]:
    """
    train_fraction: 用于控制训练集大小的比例,不完全使用全部训练数据以模拟小样本场景。
    加载并预处理 RadioML 2018.01A 数据集 (.hdf5)。
    返回: (train_data, test_data, class_map)
    其中 train_data 和 test_data 均为 (X, y, snr) 的元组。
    """
    logging.info(f"Loading data from {path}...")

    try:
        h5f = h5py.File(path, 'r')
    except ImportError:
        raise ImportError(
            "Please install h5py to load HDF5 datasets: pip install h5py")
    except Exception as e:
        raise Exception(f"Failed to open dataset: {e}")

    # 读取所有数据
    logging.info("Reading SNR (Z), Signal (X) and Label (Y) data...")
    snr = h5f['Z'][:]  # (N, 1)
    X = h5f['X'][:]    # (N, 1024, 2)
    Y = h5f['Y'][:]    # (N, 24)
    h5f.close()

    # 数据形状转换
    # 原始 X: (N, 1024, 2) -> 目标: (N, 2, 1024)
    X = np.transpose(X, (0, 2, 1))

    # 标签转换: One-hot (N, 24) -> Index (N,)
    y = np.argmax(Y, axis=1)
    # SNR: (N, 1) -> (N,)
    snr = snr.flatten()

    # 定义 RML2018 类名
    mod_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                   '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                   '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                   'FM', 'GMSK', 'OQPSK']
    class_map = {mod: i for i, mod in enumerate(mod_classes)}

    # 转为 Tensor
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    snr = torch.from_numpy(snr).float()

    # 信号功率归一化 (Average Power Normalization)
    # 计算每个样本的平均功率 P = mean(I^2 + Q^2)
    # X shape: (N, 2, 1024)
    logging.info("Normalizing signal power...")
    # dim=1 是通道维度(I,Q)，dim=2 是时间维度(1024)
    # 我们希望对每个样本(N)计算一个标量功率值
    # power shape: (N, 1, 1)
    power = torch.mean(X**2, dim=(1, 2), keepdim=True) * 2
    # 加上极小值防止除以零
    X = X / torch.sqrt(power + 1e-8)

    logging.info(f"Data loaded. Shape: {X.shape}")

    # 划分训练集/测试集
    # 根据数据集结构直接计算索引范围进行分层采样
    # 数据集结构：24种调制 x 26种信噪比 x 4096个样本
    # 顺序：先按调制方式排列，内部按信噪比从小到大排列 (-20dB 到 30dB)
    logging.info("Performing stratified split based on dataset structure...")

    n_mods = 24
    n_snrs = 26
    n_samples_per_group = 4096

    train_indices_list = []
    test_indices_list = []

    np.random.seed(42)  # 固定随机种子

    # 总共有 24 * 26 = 624 个 (Modulation, SNR) 组合块
    total_groups = n_mods * n_snrs

    # 优化：预先生成一个组内的随机索引序列，然后在循环中通过偏移量复用
    # 这样避免了在循环中重复调用 624 次 shuffle，显著提高速度
    base_indices = np.random.permutation(n_samples_per_group)
    assert CONFIG["train_fraction"] + CONFIG["test_fraction"] <= 1.0
    split_idx1 = int(n_samples_per_group * train_fraction)
    split_idx2 = int(n_samples_per_group * CONFIG["test_fraction"])
    # TODO: add validation set later
    base_train_indices = base_indices[:split_idx1]
    base_test_indices = base_indices[-split_idx2:]

    # SNR 范围是 -20dB 到 30dB，步长 2dB。
    # 动态计算 SNR 索引范围
    # SNR = -20 + index * 2  =>  index = (SNR + 20) / 2
    snr_start_idx = int((CONFIG["data_min_snr"] + 20) / 2)
    snr_end_idx = int((CONFIG["data_max_snr"] + 20) / 2)

    logging.info(
        f"Filtering SNR range: {CONFIG['data_min_snr']}dB to {CONFIG['data_max_snr']}dB (Indices: {snr_start_idx}-{snr_end_idx})")

    for i in range(total_groups):
        # 计算当前组对应的 SNR 索引
        current_snr_idx = i % n_snrs

        if snr_start_idx <= current_snr_idx <= snr_end_idx:
            offset = i * n_samples_per_group
            train_indices_list.append(base_train_indices + offset)
            test_indices_list.append(base_test_indices + offset)

    train_indices = np.concatenate(train_indices_list)
    test_indices = np.concatenate(test_indices_list)

    # 全局打乱
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # 返回 Tensor 元组，而不是 DataLoader，以便在 main.py 中灵活组合
    X_train, y_train, snr_train = X[train_indices], y[train_indices], snr[train_indices]
    X_test, y_test, snr_test = X[test_indices], y[test_indices], snr[test_indices]

    return (X_train, y_train, snr_train), (X_test, y_test, snr_test), class_map
