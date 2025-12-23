# trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging
import os
from config import CONFIG, DEVICE
from diffusion import DiffusionProcess


def train_diffusion(model: nn.Module, dataloader: DataLoader, diffusion_process: DiffusionProcess, optimizer: torch.optim.Optimizer, epochs: int, checkpoint_dir: str = "checkpoints") -> None:
    logging.info("Start training diffusion model...")
    # os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = GradScaler()

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for signals, labels in progress_bar:
            optimizer.zero_grad()

            signals, labels = signals.to(DEVICE), labels.to(DEVICE)

            # --- Label Dropout for CFG ---
            # 生成一个与 labels 形状相同的随机掩码，概率为 p_uncond
            mask = torch.rand(labels.shape, device=DEVICE) < CONFIG["p_uncond"]
            labels[mask] = CONFIG["num_classes"]

            t = torch.randint(0, CONFIG["timesteps"],
                              (signals.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(signals)

            x_t = diffusion_process.q_sample(x_start=signals, t=t, noise=noise)

            with autocast():
                predicted_noise = model(x_t, t, labels)
                loss = F.mse_loss(noise, predicted_noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # 显存清理
        torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"diffusion_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved diffusion checkpoint to {checkpoint_path}")

    logging.info("Diffusion model training complete.")


def train_classifier(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0
    for signals, labels in dataloader:
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_classifier(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for signals, labels in dataloader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def run_classifier_pipeline(model: nn.Module, train_loader: DataLoader,
                            epochs: int, learning_rate: float, checkpoint_dir: str,
                            prefix: str) -> None:
    """
    封装分类器的训练流程, 保存了模型检查点
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc=f"Training {prefix}", leave=False):
        loss = train_classifier(
            model, train_loader, optimizer, criterion)
        logging.info(
            f"{prefix} Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{prefix}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                f"Saved {prefix} checkpoint to {checkpoint_path}")
