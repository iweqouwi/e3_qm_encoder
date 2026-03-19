"""
任务 4: 损失函数模块

提供用于量子环境编码器预训练的 Log-MSE 损失函数。

物理动机:
    电子密度场 ρ(r) 的数值范围跨越多个数量级:
      - 化学键核心区域 (BCP):           ρ ~ 0.1 ~ 1.0  a.u.
      - 非共价相互作用区域 (NCI/vdW):   ρ ~ 0.001 ~ 0.01 a.u.
      - 远离分子的空白区域:             ρ ~ 1e-5 ~ 1e-6 a.u.

    标准 MSE 损失在高密度区域（化学键）的绝对误差远大于低密度区域（NCI），
    导致模型几乎完全忽视 NCI 区域。而 NCI（非共价相互作用）恰恰是
    药物分子与蛋白质口袋相互作用的关键区域（疏水堆叠、氢键等）。

    Log-MSE 对对数空间中的误差求均方，相当于对各数量级施加近似等权重，
    强制模型同时关注 NCI 弱相互作用区域和化学键强相互作用区域。

    公式:
        L = MSE(log(|ρ_pred| + ε), log(|ρ_true| + ε))
          = (1/n) Σ [log(|ρ_pred| + ε) - log(|ρ_true| + ε)]^2

    其中 ε 是避免 log(0) 的小量，默认 1e-8。
"""

import torch
import torch.nn as nn


class LogMSELoss(nn.Module):
    """
    Log-MSE 损失函数（对数均方误差）。

    注意：模型现在应该直接输出对数密度 (Log Density)，而不是物理密度。
    本损失函数仅对真实标签 (target) 取对数，然后与模型的预测值直接计算 MSE。

    同时支持一个可调的"混合系数"：在训练初期加入一小部分标准 MSE，
    有助于稳定训练（避免极小密度值导致的梯度爆炸）。

    Args:
        eps         (float): log 计算的数值稳定小量，默认 1e-8。
        mse_weight  (float): 混合的标准 MSE 权重（0 表示纯 Log-MSE），默认 0.0。
                             若希望辅助稳定训练，可设为 0.01 ~ 0.1。
        reduction   (str):   归约方式，'mean' 或 'sum'，默认 'mean'。
    """

    def __init__(
        self,
        eps:        float = 1e-8,
        mse_weight: float = 0.0,
        reduction:  str   = "mean",
    ):
        super().__init__()
        self.eps        = eps
        self.mse_weight = mse_weight
        self.reduction  = reduction

        # 内置 MSE loss（用于可选的混合损失）
        self._mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        pred: torch.Tensor,   # 预测的对数密度，形状 [..., P] (由模型直接输出无激活函数的实数)
        target: torch.Tensor, # 真实的物理密度，与 pred 形状相同
    ) -> torch.Tensor:
        """
        计算 Log-MSE 损失（可选混合标准 MSE）。

        Args:
            pred:   模型预测的对数密度，通常为 [B, P]（batch × probe 数量）
            target: 真实物理密度标签，与 pred 形状相同

        Returns:
            标量损失值
        """
        # ── 1. Log-MSE 核心计算 ───────────────────────────────────────────
        # 仅对真实的物理目标取对数，log(|ρ| + ε) 确保数值稳定
        log_target = torch.log(target.abs() + self.eps)

        # 此时 pred 已经是模型预测的对数密度，直接相减
        diff_log = pred - log_target                   # [B, P]
        
        if self.reduction == "mean":
            log_mse = (diff_log ** 2).mean()
        elif self.reduction == "sum":
            log_mse = (diff_log ** 2).sum()
        else:
            raise ValueError(f"reduction 必须是 'mean' 或 'sum'，收到 '{self.reduction}'")

        # ── 2. 可选：混合标准 MSE（训练初期稳定梯度）───────────────────────
        if self.mse_weight > 0.0:
            # 必须将预测的对数密度还原为物理密度，才能与物理 target 计算标准 MSE
            # 引入 clamp 防御机制，避免模型初期输出极大的正数导致 exp 爆炸
            physical_pred = torch.exp(torch.clamp(pred, max=0.0))
            
            mse_loss = self._mse(physical_pred, target)
            loss = log_mse + self.mse_weight * mse_loss
        else:
            loss = log_mse

        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"eps={self.eps}, "
            f"mse_weight={self.mse_weight}, "
            f"reduction='{self.reduction}')"
        )