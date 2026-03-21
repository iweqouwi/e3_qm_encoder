"""
PaiNN 量子感知编码器 (PaiNN Quantum-Aware Encoder) — 深度升级版
================================================================

本模块实现了基于 PaiNN (Polarizable Atom Interaction Neural Network) 架构的
量子环境编码器，用于从 QM/MM 复合物体系中提取物理信息特征，
为下游 SBDD (Structure-Based Drug Design) 任务提供稳健的节点条件特征。

深度升级内容：

  §2  MMPhysicsField                  — 实时计算 QM 原子处的 MM 静电势与电场（不变）
  §3  PaiNNAtomEncoder                — PaiNN 消息传递编码器，输出各层标量/矢量轨迹（不变）
  §4  MultiHeadJKNetAggregation       — 多头层级注意力聚合 + 矢量模长描述符
  §4b ScalarGatedVectorRefinement     — 标量门控矢量特征提纯层
  §5  ContinuousFilterReadout         — 连续滤波器调制读出网络（含 LayerNorm）
  ──  PaiNNQMEncoder                  — 顶层容器

设计约束：
  - 完全弃用周期性边界条件 (PBC)，采用孤立体系距离计算
  - 隐藏层通道数基数 F 作为构造参数传入
  - SE(3) 等变性：矢量特征的所有变换严禁使用 bias 和非线性激活
  - 多项式平滑截断包络函数 (Polynomial Envelope, p=5) 替代 cosine 截断
"""

from typing import List, Dict, Optional, Tuple

import math
import ase
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.charge3net.data.layer as layer
from src.charge3net.data.layer import (
    sinc_expansion,
    cosine_cutoff,
    pad_and_stack,
    unpad_and_cat,
    ShiftedSoftplus,
)

# ──────────────────────────────────────────────────────────────────────────────
# 常量：玻尔兹曼单位制下的库仑常数 (e²/(4πε₀) in eV·Å)
# 在 QM/MM 计算中常用原子单位 (a.u.)，此处保持与上游 MM 力场一致的 SI 缩放
# 若 mm_charges 的单位为基本电荷 e，坐标单位为 Å，则 ke 的值为 14.3996 eV·Å/e²
# ──────────────────────────────────────────────────────────────────────────────
_KE = 14.3996  # 库仑常数 ke (eV·Å/e²)


# ==============================================================================
# §四  多项式平滑截断包络函数 (Polynomial Envelope, p=5)
# ==============================================================================

def polynomial_envelope(distance: torch.Tensor, cutoff: float, p: int = 5) -> torch.Tensor:
    """
    DimeNet 风格的多项式平滑截断包络函数，替代 cosine 截断。

    物理动机：
      保证密度场在截断边界 r_cut 处的值及一阶/二阶导数均平滑过渡至 0，
      避免引入不连续的物理力（梯度），提升训练稳定性。

    数学定义（p=5）：
      x = d / r_cut（归一化距离）
      当 x >= 1 时：C(d) = 0
      当 x <  1 时：
          C(d) = 1 - (p+1)(p+2)/2 · x^p + p(p+2) · x^(p+1) - p(p+1)/2 · x^(p+2)
      代入 p=5：
          C(d) = 1 - 21·x^5 + 35·x^6 - 15·x^7

    性质验证：
      C(0)     = 1         → 在原子位置处完整保留贡献
      C(r_cut) = 0         → 在截断边界处贡献归零
      C'(r_cut)= 0         → 一阶导数连续（无力的不连续跳变）
      C''(r_cut)= 0        → 二阶导数连续（无力的梯度不连续跳变）

    Args:
        distance: [E, 1] 或 [E] 标量距离
        cutoff:   截断半径 r_cut (Å)
        p:        多项式阶数，默认 5

    Returns:
        C(d): 与 distance 同形状的截断权重
    """
    # 归一化距离 x = d / r_cut
    x = distance / cutoff

    # 计算多项式系数（p=5 时: c1=21, c2=35, c3=15）
    c1 = (p + 1) * (p + 2) / 2.0   # 21
    c2 = p * (p + 2)                # 35
    c3 = p * (p + 1) / 2.0         # 15

    # 多项式包络：1 - c1·x^p + c2·x^(p+1) - c3·x^(p+2)
    envelope = 1.0 - c1 * x.pow(p) + c2 * x.pow(p + 1) - c3 * x.pow(p + 2)

    # 超出截断半径的位置归零
    envelope = torch.where(
        distance < cutoff,
        envelope,
        torch.zeros_like(envelope),
    )

    return envelope


# ==============================================================================
# §2  MM 静电物理场计算模块（不变）
# ==============================================================================

class MMPhysicsField(nn.Module):
    """
    实时计算 QM 区域各原子处的 MM 静电环境特征（§2 物理场衍生）。

    对于任意 QM 原子 i（坐标 r_i），遍历所有 MM 原子 j（坐标 r_j，点电荷 q_j），
    按库仑定律求和：

        电势 (l=0 标量):
            V_i = Σ_{j∈MM}  ke·q_j / ||r_i - r_j||

        电场 (l=1 矢量):
            E_i = Σ_{j∈MM}  ke·q_j · (r_i - r_j) / ||r_i - r_j||³

    实现方式：通过 Tensor Broadcasting 在批次维度上广播，避免显式循环。
    使用 mm_mask（由 num_mm_atoms 生成）屏蔽批次内的 padding 位置。

    Args:
        eps (float): 数值稳定小量，防止距离为零时的除零错误，默认 1e-8。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        qm_positions: torch.Tensor,   # [N_total, 3]  批次内所有 QM 原子坐标（已展平）
        mm_positions: torch.Tensor,   # [B, M_max, 3] MM 原子坐标（含 padding）
        mm_charges: torch.Tensor,     # [B, M_max]    MM 原子点电荷（含 padding）
        num_nodes: torch.Tensor,      # [B]           每个样本的 QM 原子数
        num_mm_atoms: torch.Tensor,   # [B]           每个样本的真实 MM 原子数
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            V_flat: [N_total, 1]    每个 QM 原子处的电势（标量）
            E_flat: [N_total, 3]    每个 QM 原子处的电场矢量
        """
        device = qm_positions.device
        dtype  = qm_positions.dtype
        B      = mm_positions.shape[0]   # 批次大小
        M_max  = mm_positions.shape[1]   # 批次内最大 MM 原子数（含 padding）

        # ── 构建 MM padding 掩码 ──────────────────────────────────────────────
        mm_idx  = torch.arange(M_max, device=device).unsqueeze(0)  # [1, M_max]
        mm_mask = (mm_idx < num_mm_atoms.unsqueeze(1)).to(dtype)    # [B, M_max]

        # ── 对每个样本分别计算，然后拼接 ──────────────────────────────────────
        V_list, E_list = [], []
        qm_split = torch.split(qm_positions, num_nodes.tolist(), dim=0)

        for b in range(B):
            r_i = qm_split[b]                                    # [N_b, 3]
            M_b = num_mm_atoms[b].item()
            r_j = mm_positions[b, :M_b]                          # [M_b, 3]
            q_j = mm_charges[b, :M_b]                            # [M_b]

            if M_b == 0:
                N_b = r_i.shape[0]
                V_list.append(torch.zeros(N_b, 1, device=device, dtype=dtype))
                E_list.append(torch.zeros(N_b, 3, device=device, dtype=dtype))
                continue

            # ── Broadcasting 计算位移矩阵 ──────────────────────────────────
            diff_ij = r_i[:, None, :] - r_j[None, :, :]          # [N_b, M_b, 3]
            dist_ij = torch.linalg.norm(diff_ij, dim=-1)          # [N_b, M_b]
            dist_safe = dist_ij + self.eps                         # 防除零

            # ── 电势 V_i = Σ_j  ke·q_j / ||r_i - r_j|| ──────────────────
            V_contrib = _KE * q_j[None, :] / dist_safe            # [N_b, M_b]
            V_i = V_contrib.sum(dim=1, keepdim=True)               # [N_b, 1]

            # ── 电场 E_i = Σ_j  ke·q_j·(r_i - r_j) / ||r_i - r_j||³ ────
            E_contrib = (
                _KE * q_j[None, :, None] * diff_ij
                / (dist_safe ** 3).unsqueeze(-1)
            )                                                       # [N_b, M_b, 3]
            E_contrib = torch.clamp(E_contrib, min=-100.0, max=100.0)
            E_i = E_contrib.sum(dim=1)                             # [N_b, 3]

            V_list.append(V_i)
            E_list.append(E_i)

        V_flat = torch.cat(V_list, dim=0)   # [N_total, 1]
        E_flat = torch.cat(E_list, dim=0)   # [N_total, 3]
        return V_flat, E_flat


# ==============================================================================
# §3  PaiNN 原子编码器（不变）
# ==============================================================================

class PaiNNAtomEncoder(nn.Module):
    """
    PaiNN 消息传递原子编码器（§3 消息传递与特征轨迹收集）。

    严格遵循设计蓝图：
      - 标量特征维度恒为 F（由外部传入）
      - 矢量特征维度恒为 F（每个 channel 对应一个 3D 矢量，存储为 [N, 3, F]）
      - 初始化嵌入融合了：原子类别 Embedding + MM 电势投影 + MM 电场投影
      - 每层均收集 (scalar, vector) 输出，构成 L 层特征轨迹

    Args:
        num_interactions      (int):   消息传递层数 L
        hidden_size           (int):   隐藏层通道数基数 F
        cutoff                (float): QM-QM 消息截断半径（Å）
        distance_embedding_size (int): sinc RBF 展开维度
    """

    def __init__(
        self,
        num_interactions:       int,
        hidden_size:            int,
        cutoff:                 float,
        distance_embedding_size: int,
    ):
        super().__init__()
        self.num_interactions        = num_interactions
        self.hidden_size             = hidden_size
        self.cutoff                  = cutoff
        self.distance_embedding_size = distance_embedding_size
        F_dim = hidden_size

        # ── 原子类别嵌入 ────────────────────────────────────────────────────
        self.atom_embedding = nn.Embedding(
            len(ase.data.atomic_numbers), F_dim
        )

        # ── MM 电势投影（标量，可用 bias）────────────────────────────────────
        self.potential_proj = nn.Linear(1, F_dim, bias=True)

        # ── 标量初始融合 MLP ─────────────────────────────────────────────────
        self.scalar_init_mlp = nn.Sequential(
            nn.Linear(2 * F_dim, F_dim),
            nn.SiLU(),
            nn.Linear(F_dim, F_dim),
        )

        # ── MM 电场矢量投影（无 bias，保持等变性）────────────────────────────
        self.vector_init_proj = nn.Linear(1, F_dim, bias=False)

        # ── PaiNN 消息传递层（双向 QM-QM）───────────────────────────────────
        self.interactions = nn.ModuleList([
            layer.PaiNNInteraction(F_dim, distance_embedding_size, cutoff)
            for _ in range(num_interactions)
        ])

        # ── PaiNN 更新层（标量-矢量交叉更新）───────────────────────────────
        self.updates = nn.ModuleList([
            layer.PaiNNUpdate(F_dim)
            for _ in range(num_interactions)
        ])

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        V_flat: torch.Tensor,   # [N_total, 1]  MM 电势
        E_flat: torch.Tensor,   # [N_total, 3]  MM 电场矢量
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        执行 L 层 PaiNN 消息传递，收集每层输出的特征轨迹。

        Returns:
            scalar_traj: List[Tensor]  长度 L，每项 shape [N_total, F]
            vector_traj: List[Tensor]  长度 L，每项 shape [N_total, 3, F]
        """
        # ── 展平边与节点 ────────────────────────────────────────────────────
        edges_displacement = unpad_and_cat(
            input_dict["atom_edges_displacement"],
            input_dict["num_atom_edges"]
        )

        edge_offset = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=input_dict["num_nodes"].device, dtype=torch.long),
                input_dict["num_nodes"][:-1],
            ]),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = unpad_and_cat(edges, input_dict["num_atom_edges"])

        # ── 节点坐标（展平）────────────────────────────────────────────────
        atom_xyz = unpad_and_cat(
            input_dict["atom_xyz"], input_dict["num_nodes"]
        )

        # ── 计算边的欧氏距离 ──────────────────────────────────────────────
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # ── sinc RBF 展开距离 ─────────────────────────────────────────────
        edge_state = sinc_expansion(
            edges_distance,
            [(self.distance_embedding_size, self.cutoff)]
        )

        # ── 初始化节点标量特征 s_i^(0) ─────────────────────────────────────
        nodes_idx = unpad_and_cat(
            input_dict["nodes"], input_dict["num_nodes"]
        )
        s_atom = self.atom_embedding(nodes_idx)
        s_pot = self.potential_proj(V_flat)
        nodes_scalar = self.scalar_init_mlp(
            torch.cat([s_atom, s_pot], dim=-1)
        )

        # ── 初始化节点矢量特征 v_i^(0) ─────────────────────────────────────
        nodes_vector = self.vector_init_proj(
            E_flat[:, :, None]
        )

        # ── L 层 PaiNN 消息传递 + 更新，收集轨迹 ────────────────────────────
        scalar_traj: List[torch.Tensor] = []
        vector_traj: List[torch.Tensor] = []

        for interact_layer, update_layer in zip(self.interactions, self.updates):
            nodes_scalar, nodes_vector = interact_layer(
                nodes_scalar, nodes_vector,
                edge_state, edges_diff, edges_distance, edges,
            )
            nodes_scalar, nodes_vector = update_layer(
                nodes_scalar, nodes_vector
            )
            scalar_traj.append(nodes_scalar)
            vector_traj.append(nodes_vector)

        return scalar_traj, vector_traj


# ==============================================================================
# §一  多头层级注意力聚合网络 (Multi-Head JK-Net with Vector Norms)
# ==============================================================================

class MultiHeadJKNetAggregation(nn.Module):
    """
    多头 Jumping Knowledge 等变层级注意力聚合（实施文档 §一）。

    相比原版 JKNetAggregation 的核心升级：
      ① 引入矢量模长特征（旋转不变量）参与注意力计算
      ② 多头机制：将 F 维特征通道分为 H 组，每组独立学习层级选择策略
         使得网络能对短程/长程不同尺度的信息进行解耦式深度选择

    数学流程：
      1. 计算矢量模长：n_i^(l) = ||v_i^(l)||_2 ∈ R^F（旋转不变）
      2. 构建旋转不变描述符：x_i^(l) = [s_i^(l) ‖ n_i^(l)] ∈ R^{2F}
      3. MLP 计算 H 个注意力 logits：e_i^(l) = MLP_attn(x_i^(l)) ∈ R^H
      4. 沿层维度 Softmax：a_{i,h}^(l) = softmax_l(e_{i,h}^(l))
      5. 权重复制展开到 F 维：每个头控制 D=F/H 个通道
      6. 等变加权求和：
         S_JK = Σ_l â_i^(l) ⊙ s_i^(l) ∈ R^F
         V_JK = Σ_l â_i^(l) ⊙ v_i^(l) ∈ R^{3×F}

    SE(3) 等变性保证：
      - 注意力权重仅来自旋转不变量（标量 + 矢量模长）
      - 矢量通道的加权求和使用逐通道标量系数，不引入方向偏置

    Args:
        num_layers  (int): PaiNN 层数 L
        hidden_size (int): 隐藏层通道数基数 F
        num_heads   (int): 注意力头数 H（需能整除 F）
    """

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.num_heads   = num_heads

        # 验证 F 能被 H 整除
        assert hidden_size % num_heads == 0, (
            f"hidden_size={hidden_size} 必须能被 num_heads={num_heads} 整除"
        )
        self.head_dim = hidden_size // num_heads  # D = F / H

        F_dim = hidden_size

        # ── 多头注意力 MLP ──────────────────────────────────────────────────
        # 输入：旋转不变描述符 x_i^(l) = [s_i^(l) ‖ n_i^(l)] ∈ R^{2F}
        # 输出：H 个注意力 logits ∈ R^H
        # 使用两层 MLP，中间层维度 F（足够的非线性容量）
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * F_dim, F_dim),
            nn.SiLU(),
            nn.Linear(F_dim, num_heads),
        )

    def forward(
        self,
        scalar_traj: List[torch.Tensor],  # L × [N_total, F]
        vector_traj: List[torch.Tensor],  # L × [N_total, 3, F]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            S_JK: [N_total, F]    终态标量特征
            V_JK: [N_total, 3, F] 终态矢量特征
        """
        L       = self.num_layers
        H       = self.num_heads
        D       = self.head_dim
        N_total = scalar_traj[0].shape[0]
        F_dim   = self.hidden_size

        # ── Step 1：计算各层矢量模长（旋转不变量）─────────────────────────
        # v_i^(l) ∈ [N, 3, F]，沿空间维 dim=1 求范数 → n_i^(l) ∈ [N, F]
        # 物理含义：各通道的局部电场/极化强度指标
        norm_traj = []
        for v_l in vector_traj:
            # ||v||_2 沿 3D 空间维：sqrt(sum(v^2, dim=1) + eps)
            n_l = torch.sqrt((v_l ** 2).sum(dim=1) + 1e-8)  # [N, F]
            norm_traj.append(n_l)

        # ── Step 2：构建旋转不变描述符并堆叠 ──────────────────────────────
        # x_i^(l) = [s_i^(l) ‖ n_i^(l)] ∈ R^{2F}
        # 堆叠所有层：[N, L, 2F]
        x_list = []
        for s_l, n_l in zip(scalar_traj, norm_traj):
            x_l = torch.cat([s_l, n_l], dim=-1)  # [N, 2F]
            x_list.append(x_l)
        x_stack = torch.stack(x_list, dim=1)       # [N, L, 2F]

        # ── Step 3：多头注意力 logits ────────────────────────────────────
        # 将 [N, L, 2F] reshape 为 [N*L, 2F]，通过 MLP → [N*L, H]
        logits = self.attn_mlp(
            x_stack.reshape(N_total * L, -1)
        ).reshape(N_total, L, H)                    # [N, L, H]

        # ── Step 4：沿层维度 Softmax 归一化 ──────────────────────────────
        # 每个原子、每个头在 L 层上的权重之和为 1
        attn_weights = torch.softmax(logits, dim=1)  # [N, L, H]

        # ── Step 5：权重展开到 F 维（每个头复制 D 次）───────────────────
        # attn_weights: [N, L, H] → repeat_interleave → [N, L, F]
        # 每个头 h 控制 D=F/H 个连续的通道
        attn_expanded = attn_weights.repeat_interleave(D, dim=-1)  # [N, L, F]

        # ── Step 6：等变加权求和 ────────────────────────────────────────
        # 标量聚合：S_JK = Σ_l â^(l) ⊙ s^(l)
        s_stack = torch.stack(scalar_traj, dim=1)     # [N, L, F]
        S_JK = (attn_expanded * s_stack).sum(dim=1)   # [N, F]

        # 矢量聚合：V_JK = Σ_l â^(l) ⊙ v^(l)
        # attn_expanded 需在空间维广播：[N, L, 1, F] × [N, L, 3, F] → sum(dim=1) → [N, 3, F]
        # 关键 SE(3) 等变性保证：标量系数沿 3D 空间维度 (dim=2) 广播，
        # 不引入任何方向偏好，纯粹的逐通道幅度调制
        v_stack = torch.stack(vector_traj, dim=1)     # [N, L, 3, F]
        V_JK = (attn_expanded[:, :, None, :] * v_stack).sum(dim=1)  # [N, 3, F]

        return S_JK, V_JK


# ==============================================================================
# §二  标量门控矢量特征提纯层 (Scalar-Gated Vector Refinement)
# ==============================================================================

class ScalarGatedVectorRefinement(nn.Module):
    """
    标量门控矢量特征提纯层（实施文档 §二）。

    物理动机：
      JK-Net 聚合后的矢量特征 V_JK 包含多层累加的电场方向信息，
      可能混入跨层叠加的噪声。利用高阶标量特征 S_JK（包含原子电荷、
      局部配位度等上下文信息）生成 (0,1) 门控信号，动态调节矢量的强度，
      保留物理上显著的极化方向，抑制噪声。

    数学定义：
      g_i = σ(MLP_gate(S_JK,i))          ∈ R^F    门控缩放因子
      Ṽ_JK,i = g_i ⊙ V_JK,i             ∈ R^{3×F} 提纯后的矢量特征

    SE(3) 等变性保证：
      - 门控系数 g_i 完全来自旋转不变的标量特征
      - g_i 沿 3D 空间维度 (dim=1) 广播乘以矢量，不引入方向偏好
      - MLP_gate 不对矢量分量施加 bias 或非线性

    输出：
      解耦的张量元组 (S_JK, Ṽ_JK)，供下游读出网络或 SBDD 模型直接使用。

    Args:
        hidden_size (int): 隐藏层通道数基数 F
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        F_dim = hidden_size

        # ── 门控 MLP：S_JK ∈ R^F → g ∈ R^F ──────────────────────────────
        # 两层 MLP + Sigmoid 输出，生成 (0,1) 范围的通道级缩放因子
        # 第一层：F → F，非线性捕捉高阶交互
        # 第二层：F → F，线性映射到门控空间
        # Sigmoid：将任意实数压缩到 (0,1)，物理上对应"保留比例"
        self.gate_mlp = nn.Sequential(
            nn.Linear(F_dim, F_dim),
            nn.SiLU(),
            nn.Linear(F_dim, F_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        S_JK: torch.Tensor,   # [N_total, F]    终态标量特征
        V_JK: torch.Tensor,   # [N_total, 3, F] 终态矢量特征
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            S_JK:     [N_total, F]    标量特征（原样传出，不修改）
            V_JK_ref: [N_total, 3, F] 门控提纯后的矢量特征
        """
        # ── 生成门控系数 g ∈ [N, F]，值域 (0, 1) ────────────────────────
        g = self.gate_mlp(S_JK)              # [N, F]

        # ── 矢量门控调制（SE(3) 等变）──────────────────────────────────
        # g: [N, F] → [N, 1, F]，沿空间维 dim=1 广播
        # V_JK: [N, 3, F]
        # 逐元素相乘：每个 3D 矢量通道独立缩放，不改变方向
        V_JK_ref = g.unsqueeze(1) * V_JK     # [N, 3, F]

        return S_JK, V_JK_ref


# ==============================================================================
# §三  连续滤波器调制读出网络 (Continuous Filter Modulated Readout)
# ==============================================================================

class ContinuousFilterReadout(nn.Module):
    """
    连续滤波器调制读出网络（实施文档 §三）。

    核心设计哲学——严格解耦两条信息流：
      1. 距离流（绝对尺度）：RBF(d_pi) → MLP_filter → W_pi ∈ R^F
         - 描述"距离有多远"，直接决定衰减的绝对尺度
         - 不经过任何归一化，保留物理上的"绝对判决力"
      2. 潜变量流（高维原子物理）：[S_JK ‖ n_i ‖ q_pi] → Linear → LayerNorm → SiLU → Linear → state_pi ∈ R^F
         - 描述"原子是什么"，包含电荷/极化/配位等高维信息
         - 通过 LayerNorm 规范化数字分布，解决 F=196 等宽维度的方差发散问题

      两条流通过 Hadamard 乘积融合：m'_pi = W_pi ⊙ state_pi
      最终通过 MLP_out 投射到标量密度贡献：m_pi = MLP_out(m'_pi) ∈ R

    与旧 InfoBottleneckReadout 的关键区别：
      ✗ 旧版：将 e_pi (RBF) 与 S_JK, q_pi, n_i 全部 torch.cat 拼接再过 MLP
        → 距离特征与潜变量混合，LayerNorm 无法有效区分两者
      ✓ 新版：距离流和潜变量流完全独立处理，仅在最终通过乘积交互
        → LayerNorm 仅作用于潜变量，不干扰距离的绝对尺度信息

    Args:
        hidden_size             (int):   隐藏层通道数基数 F
        D_rbf                   (int):   RBF 展开维度
        cutoff                  (float): 探针-原子截断半径（Å）
        distance_embedding_size (int):   sinc RBF 的展开数 n
    """

    def __init__(
        self,
        hidden_size:            int,
        D_rbf:                  int,
        cutoff:                 float,
        distance_embedding_size: int,
    ):
        super().__init__()
        self.hidden_size             = hidden_size
        self.D_rbf                   = D_rbf
        self.cutoff                  = cutoff
        self.distance_embedding_size = distance_embedding_size
        F_dim = hidden_size

        # ══════════════════════════════════════════════════════════════════════
        # 潜变量分支 (Latent State Branch)
        # ══════════════════════════════════════════════════════════════════════
        # 输入：h_state = [S_JK ‖ n_i ‖ q_pi] ∈ R^{3F}
        # 其中：
        #   S_JK  ∈ R^F — 终态标量特征（原子身份/电荷/配位信息）
        #   n_i   ∈ R^F — 矢量模长（局部极化强度）
        #   q_pi  ∈ R^F — 方向查询内积（探针-原子方向与电场的对齐度）

        # 第一层线性变换：3F → F 降维
        self.state_linear1 = nn.Linear(3 * F_dim, F_dim)

        # LayerNorm：沿 F 维计算均值/方差并归一化
        # 解决 F=196 等宽维度下不同通道数量级差异导致的方差发散
        # 仅对潜变量施加，不影响距离流的绝对尺度
        self.state_ln = nn.LayerNorm(F_dim)

        # 第二层线性变换：F → F
        self.state_linear2 = nn.Linear(F_dim, F_dim)

        # ══════════════════════════════════════════════════════════════════════
        # 距离滤波器分支 (Distance Filter Branch)
        # ══════════════════════════════════════════════════════════════════════
        # 输入：e_pi = sinc_expand(d_pi) ∈ R^{D_rbf}
        # 输出：W_pi ∈ R^F（通道级滤波器权重）
        # 不使用任何归一化：保留距离的绝对物理尺度
        self.filter_mlp = nn.Sequential(
            nn.Linear(D_rbf, F_dim),
            nn.SiLU(),
            nn.Linear(F_dim, F_dim),
        )

        # ══════════════════════════════════════════════════════════════════════
        # 输出分支 (Output Branch)
        # ══════════════════════════════════════════════════════════════════════
        # 融合后的调制消息 m'_pi ∈ R^F → 标量密度贡献 m_pi ∈ R
        self.out_mlp = nn.Sequential(
            nn.Linear(F_dim, F_dim // 2),
            nn.SiLU(),
            nn.Linear(F_dim // 2, 1),
        )

        # 物理偏置：log(ρ_median) ≈ -6.10（对数密度经验中位数）
        self.final_bias = nn.Parameter(torch.tensor([-6.10]))

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        S_JK: torch.Tensor,   # [N_total, F]    终态标量特征
        V_JK: torch.Tensor,   # [N_total, 3, F] 门控提纯后的矢量特征
    ) -> torch.Tensor:
        """
        Returns:
            probe_output: [B, P_max]  每个批次样本的探针密度预测（含 padding）
        """
        device = S_JK.device
        dtype  = S_JK.dtype
        F_dim  = self.hidden_size

        # ══════════════════════════════════════════════════════════════════════
        # 空间几何准备：探针坐标、原子坐标、探针边
        # ══════════════════════════════════════════════════════════════════════
        atom_xyz = unpad_and_cat(
            input_dict["atom_xyz"], input_dict["num_nodes"]
        )   # [N_total, 3]

        probe_xyz = unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )   # [P_total, 3]

        # ── 构建探针边偏移 ────────────────────────────────────────────────
        atom_offset = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                input_dict["num_nodes"][:-1],
            ]),
            dim=0,
        )   # [B]

        probe_offset = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                input_dict["num_probes"][:-1],
            ]),
            dim=0,
        )   # [B]

        combined_offset = torch.stack(
            [atom_offset, probe_offset], dim=1
        )[:, None, :]   # [B, 1, 2]

        probe_edges = input_dict["probe_edges"] + combined_offset
        probe_edges = unpad_and_cat(
            probe_edges, input_dict["num_probe_edges"]
        )   # [E_probe_total, 2]  col0=atom_idx, col1=probe_idx

        # ── 计算探针-原子距离与方向向量 ──────────────────────────────────
        probe_edges_displacement = unpad_and_cat(
            input_dict["probe_edges_displacement"],
            input_dict["num_probe_edges"]
        )

        dist_pi, diff_pi = layer.calc_distance_to_probe(
            atom_xyz, probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )   # [E, 1], [E, 3]

        # ══════════════════════════════════════════════════════════════════════
        # Step 1：提取空间不变量
        # ══════════════════════════════════════════════════════════════════════

        # 获取每条边对应的原子索引
        atom_idx = probe_edges[:, 0]   # [E]

        # ── (a) RBF 距离特征 e_pi ∈ R^{D_rbf} ────────────────────────────
        e_pi = sinc_expansion(
            dist_pi,
            [(self.distance_embedding_size, self.cutoff)]
        )   # [E, D_rbf]

        # ── (b) 矢量模长 n_i ∈ R^F ──────────────────────────────────────
        # V_JK: [N, 3, F]，沿 3D 空间维求范数
        n_i = torch.sqrt((V_JK ** 2).sum(dim=1) + 1e-8)   # [N, F]
        n_e = n_i[atom_idx]                                 # [E, F]

        # ── (c) 方向性查询内积 q_pi ∈ R^F ─────────────────────────────────
        # 单位方向向量：r̂_pi = (r_p - r_i) / ||r_p - r_i||
        dist_pi_safe = torch.sqrt((diff_pi ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        r_hat_pi = diff_pi / dist_pi_safe                   # [E, 3]

        # 方向查询内积：q_pi = <r̂_pi, V_JK,i>
        # V_JK_e: [E, 3, F]，r̂_pi: [E, 3] → [E, 3, 1]
        # 沿空间维 dim=1 求和 → [E, F]（旋转不变标量）
        V_JK_e = V_JK[atom_idx]                             # [E, 3, F]
        q_pi = (V_JK_e * r_hat_pi[:, :, None]).sum(dim=1)  # [E, F]

        # ══════════════════════════════════════════════════════════════════════
        # Step 2：潜变量分支（带 LayerNorm）
        # ══════════════════════════════════════════════════════════════════════
        # 拼接潜变量描述符：h_state = [S_JK ‖ n_i ‖ q_pi] ∈ R^{3F}
        S_JK_e = S_JK[atom_idx]                              # [E, F]
        h_state = torch.cat([S_JK_e, n_e, q_pi], dim=-1)   # [E, 3F]

        # 第一层线性 + LayerNorm + SiLU + 第二层线性
        # 数学：
        #   u = W1 · h_state + b1            ∈ R^F
        #   ũ = LayerNorm(u) · γ + β         ∈ R^F（沿 F 维归一化）
        #   state = W2 · SiLU(ũ) + b2        ∈ R^F
        u = self.state_linear1(h_state)                       # [E, F]
        u_norm = self.state_ln(u)                             # [E, F] LayerNorm 归一化
        state_pi = self.state_linear2(torch.nn.functional.silu(u_norm))  # [E, F]

        # ══════════════════════════════════════════════════════════════════════
        # Step 3：距离滤波器分支（不归一化）
        # ══════════════════════════════════════════════════════════════════════
        # e_pi ∈ R^{D_rbf} → MLP_filter → W_pi ∈ R^F
        # 注意：此分支不使用 LayerNorm，保留距离的绝对物理尺度
        W_pi = self.filter_mlp(e_pi)                          # [E, F]

        # ══════════════════════════════════════════════════════════════════════
        # Step 4：物理调制融合 + 输出
        # ══════════════════════════════════════════════════════════════════════
        # 滤波器权重 × 潜变量状态（Hadamard 乘积）
        # 物理含义：距离决定"能看多远"，潜变量决定"看到什么"
        m_prime = W_pi * state_pi                             # [E, F]

        # 投射到标量密度贡献
        m_pi = self.out_mlp(m_prime).squeeze(-1)              # [E]

        # ══════════════════════════════════════════════════════════════════════
        # Step 5：多项式包络加权聚合（§四）
        # ══════════════════════════════════════════════════════════════════════
        # 使用 p=5 多项式包络替代 cosine 截断
        # C(d) = 1 - 21x^5 + 35x^6 - 15x^7，x = d/r_cut
        # 在 r_cut 处值、一阶导数、二阶导数均为 0
        envelope_weight = polynomial_envelope(
            dist_pi, self.cutoff, p=5
        ).squeeze(-1)                                          # [E]

        m_weighted = m_pi * envelope_weight                    # [E]

        # ── scatter add 聚合到探针 ─────────────────────────────────────────
        probe_idx = probe_edges[:, 1]                          # [E]
        P_total   = probe_xyz.shape[0]
        rho_flat  = torch.zeros(P_total, device=device, dtype=dtype)
        rho_flat.index_add_(0, probe_idx, m_weighted)          # [P_total]

        # 加上物理偏置（对数密度经验中位数）
        rho_flat = rho_flat + self.final_bias

        # ── 重新 pad & stack 为批次格式 ─────────────────────────────────────
        probe_output = pad_and_stack(
            torch.split(
                rho_flat,
                input_dict["num_probes"].tolist(),
                dim=0,
            )
        )   # [B, P_max]

        return probe_output


# ==============================================================================
# 顶层容器：PaiNN 量子感知密度模型（深度升级版）
# ==============================================================================

class PaiNNQMEncoder(nn.Module):
    """
    PaiNN 量子感知编码器顶层容器（深度升级版）。

    整合全部子模块，完成从 QM/MM 输入到探针密度预测的完整前向传播，
    并通过 `get_jk_features()` 将 JK 终态特征作为 API 暴露给下游 SBDD 模型。

    升级版子模块：
        mm_physics       — §2   MMPhysicsField（不变）
        atom_encoder     — §3   PaiNNAtomEncoder（不变）
        jk_aggregator    — §一  MultiHeadJKNetAggregation（多头 + 矢量模长）
        vector_refiner   — §二  ScalarGatedVectorRefinement（门控提纯）
        readout          — §三  ContinuousFilterReadout（连续滤波器 + LayerNorm）
        截断函数         — §四  polynomial_envelope（多项式包络 p=5）

    Args:
        num_interactions        (int):   PaiNN 消息传递层数 L
        hidden_size             (int):   隐藏层通道数基数 F
        cutoff                  (float): QM-QM / 探针-原子截断半径（Å）
        distance_embedding_size (int):   sinc RBF 展开维度（默认 = F // 2）
        D_rbf                   (int):   读出网络 RBF 维度（默认 = F // 2）
        num_heads               (int):   JK-Net 注意力头数 H
        mm_eps                  (float): 物理场计算数值稳定小量
    """

    def __init__(
        self,
        num_interactions:        int   = 3,
        hidden_size:             int   = 128,
        cutoff:                  float = 4.0,
        distance_embedding_size: Optional[int] = None,
        D_rbf:                   Optional[int] = None,
        num_heads:               int   = 4,
        mm_eps:                  float = 1e-8,
    ):
        super().__init__()
        self.num_interactions = num_interactions
        self.hidden_size      = hidden_size
        self.cutoff           = cutoff

        F_dim = hidden_size
        _dist_emb = distance_embedding_size if distance_embedding_size is not None else F_dim // 2
        _D_rbf    = D_rbf if D_rbf is not None else F_dim // 2

        self.distance_embedding_size = _dist_emb
        self.D_rbf = _D_rbf

        # ── §2：MM 静电物理场 ────────────────────────────────────────────
        self.mm_physics = MMPhysicsField(eps=mm_eps)

        # ── §3：PaiNN 原子编码器 ─────────────────────────────────────────
        self.atom_encoder = PaiNNAtomEncoder(
            num_interactions        = num_interactions,
            hidden_size             = F_dim,
            cutoff                  = cutoff,
            distance_embedding_size = _dist_emb,
        )

        # ── §一：多头 JK-Net 聚合（升级版）──────────────────────────────
        self.jk_aggregator = MultiHeadJKNetAggregation(
            num_layers  = num_interactions,
            hidden_size = F_dim,
            num_heads   = num_heads,
        )

        # ── §二：标量门控矢量提纯 ─────────────────────────────────────────
        self.vector_refiner = ScalarGatedVectorRefinement(
            hidden_size = F_dim,
        )

        # ── §三：连续滤波器调制读出（升级版）────────────────────────────
        self.readout = ContinuousFilterReadout(
            hidden_size             = F_dim,
            D_rbf                   = _D_rbf,
            cutoff                  = cutoff,
            distance_embedding_size = _dist_emb,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 内部辅助：运行 §2–§3–§一–§二，返回 JK 终态特征（含门控提纯）
    # ──────────────────────────────────────────────────────────────────────

    def _encode_atoms(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 §2（物理场）+ §3（PaiNN 编码）+ §一（多头 JK 聚合）+ §二（门控提纯），
        返回终态特征 (S_JK, V_JK_refined)。

        Returns:
            S_JK:         [N_total, F]    终态标量特征
            V_JK_refined: [N_total, 3, F] 门控提纯后的等变矢量特征
        """
        # §2：计算 QM 原子处的 MM 静电势与电场
        atom_xyz_flat = unpad_and_cat(
            input_dict["atom_xyz"], input_dict["num_nodes"]
        )

        V_flat, E_flat = self.mm_physics(
            qm_positions  = atom_xyz_flat,
            mm_positions  = input_dict["mm_positions"],
            mm_charges    = input_dict["mm_charges"],
            num_nodes     = input_dict["num_nodes"],
            num_mm_atoms  = input_dict["num_mm_atoms"],
        )

        # §3：PaiNN L 层消息传递，收集特征轨迹
        scalar_traj, vector_traj = self.atom_encoder(
            input_dict, V_flat, E_flat
        )

        # §一：多头 JK-Net 注意力聚合
        S_JK, V_JK = self.jk_aggregator(scalar_traj, vector_traj)

        # §二：标量门控矢量提纯
        S_JK, V_JK_refined = self.vector_refiner(S_JK, V_JK)

        return S_JK, V_JK_refined

    # ──────────────────────────────────────────────────────────────────────
    # 公开 API：暴露 JK 特征给下游模型
    # ──────────────────────────────────────────────────────────────────────

    def get_jk_features(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【下游 SBDD 接口】运行 §2–§3–§一–§二，返回节点级 JK 终态条件特征。

        Returns:
            S_JK:         [N_total, F]    终态标量特征（旋转不变）
            V_JK_refined: [N_total, 3, F] 门控提纯后的等变矢量特征（SE(3) 等变）
        """
        return self._encode_atoms(input_dict)

    # ──────────────────────────────────────────────────────────────────────
    # 完整前向传播：用于预训练密度预测任务
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        完整前向传播：QM/MM 输入 → 探针密度预测。

        Args:
            input_dict: 图字典，包含以下键：
                atom_xyz               [B, N_max, 3]
                nodes                  [B, N_max]      原子序数（int64）
                atom_edges             [B, E_max, 2]
                atom_edges_displacement[B, E_max, 3]
                num_nodes              [B]
                num_atom_edges         [B]
                probe_xyz              [B, P_max, 3]
                probe_edges            [B, Ep_max, 2]
                probe_edges_displacement[B, Ep_max, 3]
                num_probes             [B]
                num_probe_edges        [B]
                cell                   [B, 3, 3]
                mm_positions           [B, M_max, 3]
                mm_charges             [B, M_max]
                num_mm_atoms           [B]

        Returns:
            probe_output: [B, P_max]  各探针处的预测对数密度
        """
        # §2–§3–§一–§二：原子编码 + JK 聚合 + 门控提纯
        S_JK, V_JK_refined = self._encode_atoms(input_dict)

        # §三：连续滤波器调制读出 + §四：多项式包络截断
        probe_output = self.readout(input_dict, S_JK, V_JK_refined)

        return probe_output

    # ──────────────────────────────────────────────────────────────────────
    # 分步接口：供 Trainer._test_step 缓存 atom_repr 复用
    # ──────────────────────────────────────────────────────────────────────

    def atom_model(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅执行原子编码, 返回 (S_JK, V_JK_refined), 可以直接供下游作为特征输入。
        命名与旧 PainnDensityModel.atom_model 对齐，方便 Trainer 复用。
        """
        return self._encode_atoms(input_dict)

    def probe_model(
        self,
        input_dict: Dict[str, torch.Tensor],
        S_JK: torch.Tensor,
        V_JK: torch.Tensor,
    ) -> torch.Tensor:
        """
        仅执行探针读出（§三），复用已缓存的原子特征。
        命名与旧 PainnDensityModel.probe_model 对齐，方便 Trainer 复用。
        """
        return self.readout(input_dict, S_JK, V_JK)
