# main.py

import torch
import torch.nn as nn
import os
import logging
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG, DEVICE
from data_loader import load_and_preprocess_data
from models import ConditionalUNet1D, CNNClassifier
from diffusion import DiffusionProcess
from trainer import train_diffusion, run_classifier_pipeline, evaluate_classifier
from utils import generate_augmented_dataset


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("training.log", mode='a', encoding='utf-8'),
            # logging.StreamHandler()
        ]
    )
    logging.info("="*30)
    logging.info("Starting main process...")

    # 创建带有时间戳的检查点目录，防止覆盖
    timestamp = datetime.now().strftime("%m-%d--%H-%M")
    run_checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    logging.info(
        f"Checkpoints for this run will be saved to: {run_checkpoint_dir}")

    # Step A: 加载数据
    train_frac = CONFIG['train_fraction']

    # 获取原始 Tensor 数据
    (X_train, y_train, snr_train), (X_test, y_test, snr_test), class_map = load_and_preprocess_data(
        CONFIG["data_path"], train_fraction=train_frac)

    # 1. 准备扩散模型训练数据：仅使用高信噪比 (SNR >= x dB)
    high_snr_mask = snr_train >= CONFIG["diffusion_snr_start"]
    diffusion_train_dataset = TensorDataset(
        X_train[high_snr_mask], y_train[high_snr_mask])
    diffusion_train_loader = DataLoader(
        diffusion_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    logging.info(
        f"Diffusion Training Data (SNR>={CONFIG['diffusion_snr_start']} dB): {len(diffusion_train_dataset)} samples")

    # 2. 准备基线分类器训练数据：使用全量数据 (SNR in [min_snr, max_snr])
    baseline_train_dataset = TensorDataset(X_train, y_train)
    baseline_train_loader = DataLoader(
        baseline_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    logging.info(
        f"Baseline Classifier Training Data SNR in [{CONFIG['data_min_snr']}, {CONFIG['data_max_snr']}]: {len(baseline_train_dataset)} samples")

    # 3. 准备测试数据：使用全量数据
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Step B: 训练或加载扩散模型
    diffusion_model_path = CONFIG["diffusion_model_path"]
    diffusion_model = ConditionalUNet1D().to(DEVICE)
    # diffusion_model_path = "/root/code/DM_AMR/checkpoints/11-30--13-41/diffusion_epoch_10.pth"
    # run_checkpoint_dir = "/root/code/DM_AMR/checkpoints/11-30--13-41/"
    baseline_classifier_path = None
    augmented_classifier_path = None
    # TODO: add classifier path loading later

    if diffusion_model_path is None:
        diffusion_optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=CONFIG["learning_rate"])
        diffusion_process = DiffusionProcess()
        train_diffusion(diffusion_model, diffusion_train_loader,
                        diffusion_process, diffusion_optimizer, CONFIG["epochs_diffusion"], checkpoint_dir=run_checkpoint_dir)
    else:
        logging.info(
            f"Loading pre-trained diffusion model from {diffusion_model_path}")
        diffusion_model.load_state_dict(torch.load(
            diffusion_model_path, map_location=DEVICE))

    # Step C: 训练并评估基线分类器
    logging.info("\n--- Training Baseline Classifier (on original data) ---")
    baseline_classifier = CNNClassifier().to(DEVICE)
    if baseline_classifier_path:
        logging.info(
            f"Loading pre-trained baseline classifier from {baseline_classifier_path}")
        baseline_classifier.load_state_dict(torch.load(
            baseline_classifier_path, map_location=DEVICE))
    else:
        run_classifier_pipeline(
            baseline_classifier,
            baseline_train_loader,
            CONFIG["epochs_classifier"],
            CONFIG["learning_rate"],
            run_checkpoint_dir,
            "baseline_classifier"
        )
    baseline_accuracy = evaluate_classifier(baseline_classifier, test_loader)
    logging.info(f"Baseline Classifier Accuracy: {baseline_accuracy:.2f}%")

    # Step D: 生成、筛选并创建增强数据集
    all_gen_signals, all_gen_labels = generate_augmented_dataset(
        diffusion_model, baseline_classifier, class_map, run_checkpoint_dir)

    augmented_signals = torch.cat([X_train, all_gen_signals])
    augmented_labels = torch.cat([y_train, all_gen_labels])
    augmented_loader = DataLoader(TensorDataset(
        augmented_signals, augmented_labels), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    # Step E: 训练并评估增强后的分类器
    logging.info("\n--- Training Augmented Classifier ---")
    augmented_classifier = CNNClassifier().to(DEVICE)
    if augmented_classifier_path:
        logging.info(
            f"Loading pre-trained augmented classifier from {augmented_classifier_path}")
        augmented_classifier.load_state_dict(torch.load(
            augmented_classifier_path, map_location=DEVICE))
    else:
        run_classifier_pipeline(
            augmented_classifier,
            augmented_loader,
            CONFIG["epochs_classifier"],
            CONFIG["learning_rate"],
            run_checkpoint_dir,
            "augmented_classifier"
        )
    augmented_accuracy = evaluate_classifier(augmented_classifier, test_loader)
    logging.info(f"Augmented Classifier Accuracy: {augmented_accuracy:.2f}%")

    # Step F: 结果对比
    logging.info("\n" + "="*30)
    logging.info("---           FINAL RESULTS           ---")
    logging.info("="*30)
    logging.info(f"Dataset: RadioML 2018.01A (SNR >= x dB)")
    logging.info(f"Real Training Data Fraction: {train_frac * 100:.1f}%")
    logging.info("-" * 30)
    logging.info(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    logging.info(f"Augmented Accuracy: {augmented_accuracy:.2f}%")
    logging.info(
        f"Performance Improvement: {augmented_accuracy - baseline_accuracy:.2f}%")
    logging.info("="*30)


if __name__ == '__main__':
    main()
