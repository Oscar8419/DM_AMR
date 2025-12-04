import torch
import torch.nn.functional as F
import os
import logging
import numpy as np
from tqdm import tqdm

from config import CONFIG, DEVICE
from diffusion import DiffusionProcess


def generate_augmented_dataset(diffusion_model, baseline_classifier, class_map, run_checkpoint_dir):
    """
    生成并筛选增强数据集
    """
    logging.info(
        "\n--- Generating and Filtering New Samples for Augmentation ---")
    diffusion_process = DiffusionProcess()
    generated_signals, generated_labels = [], []

    # Ensure models are in eval mode for inference
    diffusion_model.eval()
    baseline_classifier.eval()

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

    else:
        logging.warning("No samples were generated/filtered!")
        all_gen_signals = torch.empty(0, 2, 1024)
        all_gen_labels = torch.empty(0, dtype=torch.long)

    return all_gen_signals, all_gen_labels
