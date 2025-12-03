# diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
from config import CONFIG, DEVICE


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)


class DiffusionProcess:
    def __init__(self, timesteps: int = CONFIG["timesteps"]):
        self.timesteps = timesteps

        # 1. 定义噪声调度 beta_t
        self.betas = linear_beta_schedule(timesteps)

        # 2. 定义 alpha_t = 1 - beta_t
        self.alphas = 1. - self.betas

        # 3. 计算 alpha 的累积乘积 (alpha_bar_t)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # 4. 计算前一时刻的 alpha_bar_{t-1} (用于后验计算)
        # 通过右移一位并填充 1.0 来实现 (t=0 时 prev 为 1.0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 5. 预计算前向扩散过程 q(x_t | x_0) 所需的系数
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)

        # 6. 计算后验方差 (posterior variance) tilde_beta_t
        # 用于反向采样过程 p(x_{t-1} | x_t)
        # var = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        # 根据时间步 t 选择对应的系数. 注意广播机制
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t, None, None]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t_tensor: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        betas_t = self.betas[t_tensor, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[
            t_tensor, None, None]
        sqrt_recip_alphas_t = torch.sqrt(
            1.0 / self.alphas[t_tensor, None, None])

        predicted_noise = model(x, t_tensor, classes)
        model_mean = sqrt_recip_alphas_t * \
            (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_tensor[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_tensor, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_ddpm(self, model: nn.Module, num_samples: int, classes: torch.Tensor) -> torch.Tensor:
        shape = (num_samples, 2, CONFIG["signal_length"])
        img = torch.randn(shape, device=DEVICE)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps, leave=False):
            t_tensor = torch.full(
                (num_samples,), i, device=DEVICE, dtype=torch.long)
            img = self.p_sample(model, img, t_tensor, classes)
        return img

    @torch.no_grad()
    def sample_ddim(self, model: nn.Module, num_samples: int, classes: torch.Tensor, ddim_timesteps: int = CONFIG["ddim_timesteps"], eta: float = CONFIG["ddim_eta"]) -> torch.Tensor:
        """
        DDIM Sampling.
        Args:
            ddim_timesteps: Number of steps for DDIM sampling (e.g., 50 or 100).
            eta: 0.0 for deterministic DDIM, 1.0 for DDPM-like stochasticity.
        """
        shape = (num_samples, 2, CONFIG["signal_length"])
        img = torch.randn(shape, device=DEVICE)

        # 生成时间步序列 (e.g., [0, 20, 40, ..., 980])
        # 使用简单的均匀间隔采样
        c = self.timesteps // ddim_timesteps
        time_steps = list(range(0, self.timesteps, c)) + [self.timesteps - 1]
        # 确保只取 ddim_timesteps 个点，并反转顺序
        time_steps = time_steps[:ddim_timesteps]
        time_steps = list(reversed(time_steps))

        for i in tqdm(range(len(time_steps)), desc="DDIM Sampling", total=len(time_steps), leave=False):
            t = time_steps[i]
            prev_t = time_steps[i+1] if i < len(time_steps) - 1 else -1

            t_tensor = torch.full(
                (num_samples,), t, device=DEVICE, dtype=torch.long)

            # 1. 预测噪声
            noise_pred = model(img, t_tensor, classes)

            # 2. 计算 alpha_bar
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(
                1.0, device=DEVICE)

            # 3. 预测 x0 (denoised)
            # x0 = (xt - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
            pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) *
                       noise_pred) / torch.sqrt(alpha_bar_t)

            # 4. 计算方向指向 xt (direction pointing to xt)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) /
                                       (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * noise_pred

            # 5. 更新 x (x_{t-1})
            noise = torch.randn_like(img) if eta > 0 else 0.
            img = torch.sqrt(alpha_bar_prev) * pred_x0 + \
                dir_xt + sigma_t * noise

        return img
