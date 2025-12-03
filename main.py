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
from trainer import train_diffusion, train_classifier, evaluate_classifier


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
    diffusion_model_path = "/root/code/DM_AMR/checkpoints/11-30--13-41/diffusion_epoch_10.pth"
    run_checkpoint_dir = "/root/code/DM_AMR/checkpoints/11-30--13-41/"

    if not os.path.exists(diffusion_model_path):
        diffusion_optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=CONFIG["learning_rate"])
        diffusion_process = DiffusionProcess()
        train_diffusion(diffusion_model, diffusion_train_loader,
                        diffusion_process, diffusion_optimizer, CONFIG["epochs_diffusion"], checkpoint_dir=run_checkpoint_dir)
        torch.save(diffusion_model.state_dict(), diffusion_model_path)
    else:
        logging.info(
            f"Loading pre-trained diffusion model from {diffusion_model_path}")
        diffusion_model.load_state_dict(torch.load(
            diffusion_model_path, map_location=DEVICE))

    # Step C: 训练并评估基线分类器
    logging.info("\n--- Training Baseline Classifier (on original data) ---")
    baseline_classifier = CNNClassifier().to(DEVICE)
    baseline_optimizer = torch.optim.Adam(
        baseline_classifier.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(CONFIG["epochs_classifier"]), desc="Training Baseline Classifier", leave=False):
        loss = train_classifier(
            baseline_classifier, baseline_train_loader, baseline_optimizer, criterion)
        logging.info(
            f"Baseline Epoch {epoch+1}/{CONFIG['epochs_classifier']}, Loss: {loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == CONFIG["epochs_classifier"]:
            checkpoint_path = os.path.join(
                run_checkpoint_dir, f"baseline_classifier_epoch_{epoch+1}.pth")
            torch.save(baseline_classifier.state_dict(), checkpoint_path)
            logging.info(
                f"Saved baseline classifier checkpoint to {checkpoint_path}")
    baseline_accuracy = evaluate_classifier(baseline_classifier, test_loader)

    # Step D: 生成、筛选并创建增强数据集
    logging.info(
        "\n--- Generating and Filtering New Samples for Augmentation ---")
    diffusion_process = DiffusionProcess()
    generated_signals, generated_labels = [], []

    for class_idx, class_name in enumerate(sorted(class_map, key=class_map.get)):
        logging.info(
            f"Generating samples for class: {class_name} ({class_idx})")

        # 分批生成以避免显存溢出 (OOM)
        total_samples = CONFIG["num_generated_samples_per_class"]
        gen_batch_size = CONFIG["diffusion_gene_batch_size"]
        num_batches = int(np.ceil(total_samples / gen_batch_size))
        class_filtered_count = 0

        current_class_signals = []  # 暂存当前类别的生成结果

        # 使用 tqdm 添加生成进度条
        for i in tqdm(range(num_batches), desc=f"Generating for {class_name}"):

            # 1. 构造条件标签张量
            labels_to_gen = torch.full(
                (gen_batch_size,), class_idx, device=DEVICE, dtype=torch.long)

            # 2. 执行反向扩散采样, ddim OR ddpm
            batch_samples = diffusion_process.sample_ddim(
                diffusion_model, gen_batch_size, labels_to_gen)

            with torch.no_grad():
                preds = baseline_classifier(batch_samples)
                probs = F.softmax(preds, dim=1)
                confidences, pred_classes = torch.max(probs, 1)
                mask = (pred_classes == class_idx) & (
                    confidences >= CONFIG["confidence_threshold"])
                filtered_samples = batch_samples[mask]

                if len(filtered_samples) > 0:
                    current_class_signals.append(filtered_samples.cpu())
                    class_filtered_count += len(filtered_samples)

        logging.info(
            f"  - Generated {total_samples}, Filtered to {class_filtered_count} high-confidence samples.")

        # 保存当前类别的样本
        if len(current_class_signals) > 0:
            class_signals_tensor = torch.cat(current_class_signals, dim=0)
            class_labels_tensor = torch.full(
                (len(class_signals_tensor),), class_idx, dtype=torch.long)

            class_save_path = os.path.join(
                run_checkpoint_dir, f"generated_class_{class_idx}_{class_name}.pt")
            torch.save({
                'signals': class_signals_tensor,
                'labels': class_labels_tensor
            }, class_save_path)

            # 添加到总列表
            generated_signals.extend(current_class_signals)
            generated_labels.extend([class_idx] * len(class_signals_tensor))

    # 保存生成的样本到磁盘
    if len(generated_signals) > 0:
        # 拼接所有生成的信号
        all_gen_signals = torch.cat(generated_signals, dim=0)
        all_gen_labels = torch.tensor(generated_labels, dtype=torch.long)

        # 1. 保存为 PyTorch .pt 格式 (方便后续加载训练)
        pt_save_path = os.path.join(run_checkpoint_dir, "generated_samples.pt")
        torch.save({
            'signals': all_gen_signals,
            'labels': all_gen_labels
        }, pt_save_path)
        logging.info(
            f"Saved {len(all_gen_signals)} generated samples to {pt_save_path}")

        # 2. 保存为 NumPy .npz 格式 (方便画图分析)
        # np_save_path = os.path.join(
        #     run_checkpoint_dir, "generated_samples.npz")
        # np.savez_compressed(np_save_path,
        #                     signals=all_gen_signals.numpy(),
        #                     labels=all_gen_labels.numpy())
        # logging.info(f"Saved NumPy format to {np_save_path}")
    else:
        logging.warning("No samples were generated/filtered!")
        all_gen_signals = torch.empty(0, 2, 1024)
        all_gen_labels = torch.empty(0, dtype=torch.long)

    augmented_signals = torch.cat([X_train, all_gen_signals])
    augmented_labels = torch.cat([y_train, all_gen_labels])
    augmented_loader = DataLoader(TensorDataset(
        augmented_signals, augmented_labels), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    # Step E: 训练并评估增强后的分类器
    logging.info("\n--- Training Augmented Classifier ---")
    augmented_classifier = CNNClassifier().to(DEVICE)
    augmented_optimizer = torch.optim.Adam(
        augmented_classifier.parameters(), lr=CONFIG["learning_rate"])

    for epoch in tqdm(range(CONFIG["epochs_classifier"]), desc="Training Augmented Classifier", leave=False):
        loss = train_classifier(
            augmented_classifier, augmented_loader, augmented_optimizer, criterion)
        logging.info(
            f"Augmented Epoch {epoch+1}/{CONFIG['epochs_classifier']}, Loss: {loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == CONFIG["epochs_classifier"]:
            checkpoint_path = os.path.join(
                run_checkpoint_dir, f"augmented_classifier_epoch_{epoch+1}.pth")
            torch.save(augmented_classifier.state_dict(), checkpoint_path)
            logging.info(
                f"Saved augmented classifier checkpoint to {checkpoint_path}")

    augmented_accuracy = evaluate_classifier(augmented_classifier, test_loader)

    # Step F: 结果对比
    logging.info("\n" + "="*30)
    logging.info("---           FINAL RESULTS           ---")
    logging.info("="*30)
    logging.info(f"Dataset: RadioML 2018.01A (SNR >= 0dB)")
    logging.info(f"Real Training Data Fraction: {train_frac * 100:.1f}%")
    logging.info("-" * 30)
    logging.info(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    logging.info(f"Augmented Accuracy: {augmented_accuracy:.2f}%")
    logging.info(
        f"Performance Improvement: {augmented_accuracy - baseline_accuracy:.2f}%")
    logging.info("="*30)


if __name__ == '__main__':
    main()
