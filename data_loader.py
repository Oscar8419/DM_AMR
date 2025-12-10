# data_loader.py

import torch
import numpy as np
import h5py
import logging
from typing import Tuple, Dict
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

    # 定义 RML2018 类名
    mod_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                   '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                   '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                   'FM', 'GMSK', 'OQPSK']
    class_map = {mod: i for i, mod in enumerate(mod_classes)}

    logging.info("Processing data in chunks to save memory...")

    n_mods = 24
    n_snrs = 26
    n_samples_per_group = 4096

    # 预先计算好每个 group 的随机索引
    np.random.seed(42)
    base_indices = np.random.permutation(n_samples_per_group)

    split_idx1 = int(n_samples_per_group * train_fraction)
    split_idx2 = int(n_samples_per_group * CONFIG["test_fraction"])

    train_mask_indices = base_indices[:split_idx1]
    test_mask_indices = base_indices[-split_idx2:]

    train_X_list, train_y_list, train_snr_list = [], [], []
    test_X_list, test_y_list, test_snr_list = [], [], []

    # SNR 范围是 -20dB 到 30dB，步长 2dB。
    snr_start_idx = int((CONFIG["data_min_snr"] + 20) / 2)
    snr_end_idx = int((CONFIG["data_max_snr"] + 20) / 2)
    logging.info(
        f"Filtering SNR range: {CONFIG['data_min_snr']}dB to {CONFIG['data_max_snr']}dB")

    # 循环处理每个 group
    total_groups = n_mods * n_snrs
    for i in range(total_groups):
        current_snr_idx = i % n_snrs
        if not (snr_start_idx <= current_snr_idx <= snr_end_idx):
            continue

        if i % 50 == 0:
            logging.info(f"Processing group {i}/{total_groups}...")

        start_idx = i * n_samples_per_group
        end_idx = start_idx + n_samples_per_group

        # 读取当前 group 的数据
        X_group = h5f['X'][start_idx:end_idx]  # (4096, 1024, 2)
        Y_group = h5f['Y'][start_idx:end_idx]  # (4096, 24)
        Z_group = h5f['Z'][start_idx:end_idx]  # (4096, 1)

        # 预处理
        # 1. Transpose X: (4096, 1024, 2) -> (4096, 2, 1024)
        X_group = np.transpose(X_group, (0, 2, 1))

        # 2. Process Labels
        y_group = np.argmax(Y_group, axis=1)
        snr_group = Z_group.flatten()

        # 3. Convert to Tensor
        X_t = torch.from_numpy(X_group).float()
        y_t = torch.from_numpy(y_group).long()
        snr_t = torch.from_numpy(snr_group).float()

        # 4. Normalize
        power = torch.mean(X_t**2, dim=(1, 2), keepdim=True) * 2
        X_t = X_t / torch.sqrt(power + 1e-8)

        # 5. Split and Append
        train_X_list.append(X_t[train_mask_indices])
        train_y_list.append(y_t[train_mask_indices])
        train_snr_list.append(snr_t[train_mask_indices])

        test_X_list.append(X_t[test_mask_indices])
        test_y_list.append(y_t[test_mask_indices])
        test_snr_list.append(snr_t[test_mask_indices])

    h5f.close()

    logging.info("Concatenating chunks...")
    X_train = torch.cat(train_X_list)
    y_train = torch.cat(train_y_list)
    snr_train = torch.cat(train_snr_list)

    X_test = torch.cat(test_X_list)
    y_test = torch.cat(test_y_list)
    snr_test = torch.cat(test_snr_list)

    logging.info(
        f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return (X_train, y_train, snr_train), (X_test, y_test, snr_test), class_map
