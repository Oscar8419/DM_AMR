# trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
from config import CONFIG, DEVICE
from diffusion import DiffusionProcess


def train_diffusion(model: nn.Module, dataloader: DataLoader, diffusion_process: DiffusionProcess, optimizer: torch.optim.Optimizer, epochs: int, checkpoint_dir: str = "checkpoints") -> None:
    logging.info("Start training diffusion model...")
    # os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for signals, labels in progress_bar:
            optimizer.zero_grad()

            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            t = torch.randint(0, CONFIG["timesteps"],
                              (signals.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(signals)

            x_t = diffusion_process.q_sample(x_start=signals, t=t, noise=noise)
            predicted_noise = model(x_t, t, labels)

            loss = F.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
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
