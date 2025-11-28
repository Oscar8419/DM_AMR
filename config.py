# config.py

import torch
import platform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 自动判断数据集路径
if platform.system() == 'Windows':
    DATA_PATH = "D:\\Dataset\\2018.01\\GOLD_XYZ_OSC.0001_1024.hdf5"
else:
    # Linux 服务器路径
    DATA_PATH = "/root/autodl-tmp/GOLD_XYZ_OSC.0001_1024.hdf5"

CONFIG = {
    # --- 数据与路径 ---
    "data_path": DATA_PATH,  # RML2018数据集路径
    "diffusion_model_path": "conditional_diffusion_model_2018.pth",

    # --- 模型参数 ---
    "num_classes": 24,              # RadioML2018.01A有24种调制类型
    "signal_length": 1024,          # 信号长度为1024

    # --- 扩散模型参数 ---
    "timesteps": 200,               # 扩散步数 (T)

    # --- 训练参数 ---
    "epochs_diffusion": 50,         # 训练扩散模型的轮数
    "epochs_classifier": 30,        # 训练分类器的轮数
    "batch_size": 64,               # 信号变长了，适当减小batch_size防止显存溢出
    "learning_rate": 1e-3,

    # --- 数据增强与评估参数 ---
    "confidence_threshold": 0.8,    # 筛选生成样本的置信度阈值
    # 每类生成的样本数 TODO:increase number later
    "num_generated_samples_per_class": 30 * 1000,
    # "train_data_fraction": 1.0,      # 使用的真实训练数据比例 (1.0代表全部，0.5代表50%)
    "train_fraction": 0.5,  # 每种组合4096个样本，训练集比例, 不完全使用，模拟小样本场景
    "test_fraction": 0.2,  # 每种组合4096个样本，测试集比例
    "diffusion_snr_start": 6,  # 从这个SNR开始训练扩散模型

    # --- 数据筛选参数 ---
    "data_min_snr": -10,    # 最小保留SNR (dB)
    "data_max_snr": 20,     # 最大保留SNR (dB)
}
