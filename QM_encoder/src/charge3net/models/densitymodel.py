"""
PaiNN 量子感知在编码器 (PaiNN Quantum-Aware Encoder)
====================================================

本模块实现了基于 PaiNN (Polarizable Atom Interaction Neural Network) 架构的
量子环境编码器，用于从 QM/MM 复合物体系中提取物理信息特征，
为下游 SBDD (Structure-Based Drug Design) 任务提供稳健的节点条件特征。

整体架构分为五个阶段（严格对应设计蓝图 §1–§5）：

  §2  MMPhysicsField        — 实时计算 QM 原子处的 MM 静电势与电场
  §3  PaiNNAtomEncoder      — PaiNN 消息传递编码器，输出各层标量/矢量轨迹
  §4  JKNetAggregation      — Jumping Knowledge 层级注意力聚合
  §5  InfoBottleneckReadout — 原子→探针单跳信息瓶颈读出网络
  ──  PaiNNDensityModel     — 顶层容器，暴露 JK 特征 API

设计约束：
  - 完全弃用周期性边界条件 (PBC)，采用孤立体系距离计算
  - 隐藏层通道数基数 F 作为构造参数传入（蓝图建议 F=128）
  - SE(3) 等变性：矢量特征的所有变换严禁使用 bias 和非线性激活

作者：QM_encoder 项目组
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
# §2  MM 静电物理场计算模块
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
        # mask[b, j] = 1 表示样本 b 的第 j 个 MM 原子是真实原子（非 padding）
        # Shape: [B, M_max]
        mm_idx  = torch.arange(M_max, device=device).unsqueeze(0)  # [1, M_max]
        mm_mask = (mm_idx < num_mm_atoms.unsqueeze(1)).to(dtype)    # [B, M_max]

        # ── 对每个样本分别计算，然后拼接 ──────────────────────────────────────
        # 注：QM 原子在批次维度已展平（unpad_and_cat），需按 num_nodes 分割回样本
        V_list, E_list = [], []
        qm_split = torch.split(qm_positions, num_nodes.tolist(), dim=0)

        for b in range(B):
            # r_i: [N_b, 3]  当前样本的 QM 原子坐标
            # r_j: [M_b, 3]  当前样本的 MM 原子坐标（截取真实部分）
            # q_j: [M_b]     当前样本的 MM 原子电荷
            r_i = qm_split[b]                                    # [N_b, 3]
            M_b = num_mm_atoms[b].item()
            r_j = mm_positions[b, :M_b]                          # [M_b, 3]
            q_j = mm_charges[b, :M_b]                            # [M_b]

            if M_b == 0:
                # 无 MM 原子时，物理场为零
                N_b = r_i.shape[0]
                V_list.append(torch.zeros(N_b, 1, device=device, dtype=dtype))
                E_list.append(torch.zeros(N_b, 3, device=device, dtype=dtype))
                continue

            # ── Broadcasting 计算位移矩阵 ──────────────────────────────────
            # r_i[:, None, :]  -> [N_b, 1,   3]
            # r_j[None, :, :]  -> [1,   M_b, 3]
            # diff_ij          -> [N_b, M_b, 3]  diff_ij[i,j] = r_i - r_j
            diff_ij = r_i[:, None, :] - r_j[None, :, :]          # [N_b, M_b, 3]

            # 距离 ||r_i - r_j||  Shape: [N_b, M_b]
            dist_ij = torch.linalg.norm(diff_ij, dim=-1)          # [N_b, M_b]
            dist_safe = dist_ij + self.eps                         # 防除零

            # ── 电势 V_i = Σ_j  ke·q_j / ||r_i - r_j|| ──────────────────
            # q_j[None, :] -> [1, M_b]，与 dist_safe[N_b, M_b] 广播相乘
            V_contrib = _KE * q_j[None, :] / dist_safe            # [N_b, M_b]
            V_i = V_contrib.sum(dim=1, keepdim=True)               # [N_b, 1]

            # ── 电场 E_i = Σ_j  ke·q_j·(r_i - r_j) / ||r_i - r_j||³ ────
            # dist_safe³ Shape: [N_b, M_b]
            # q_j[None, :, None] -> [1, M_b, 1]，与 diff_ij[N_b, M_b, 3] 广播
            E_contrib = (
                _KE * q_j[None, :, None] * diff_ij
                / (dist_safe ** 3).unsqueeze(-1)
            )                                                       # [N_b, M_b, 3]
            E_i = E_contrib.sum(dim=1)                             # [N_b, 3]

            V_list.append(V_i)
            E_list.append(E_i)

        V_flat = torch.cat(V_list, dim=0)   # [N_total, 1]
        E_flat = torch.cat(E_list, dim=0)   # [N_total, 3]
        return V_flat, E_flat


# ==============================================================================
# §3  PaiNN 原子编码器
# ==============================================================================

class PaiNNAtomEncoder(nn.Module):
    """
    PaiNN 消息传递原子编码器（§3 消息传递与特征轨迹收集）。

    严格遵循设计蓝图：
      - 标量特征维度恒为 F（由外部传入）
      - 矢量特征维度恒为 F（每个 channel 对应一个 3D 矢量，存储为 [N, 3, F]）
      - 初始化嵌入融合了：原子类别 Embedding + MM 电势投影 + MM 电场投影
      - 每层均收集 (scalar, vector) 输出，构成 L 层特征轨迹

    初始标量特征（§2 → §3 衔接）：
        s_atom = Embedding(Z_i)                     ∈ R^F
        s_pot  = Linear_bias(V_i)                   ∈ R^F
        s_i^(0) = MLP([s_atom ‖ s_pot])             ∈ R^F

    初始矢量特征（严格等变，无 bias）：
        v_i^(0) = Linear_no_bias(E_i)               ∈ R^{F×3}
        （等价于将 1 通道电场线性投影为 F 通道，保持方向性）

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
        F = hidden_size

        # ── 原子类别嵌入 ────────────────────────────────────────────────────
        # Z_i ∈ {1,...,118}，映射到 F 维连续向量
        self.atom_embedding = nn.Embedding(
            len(ase.data.atomic_numbers), F
        )

        # ── MM 电势投影（标量，可用 bias）────────────────────────────────────
        # V_i: [N, 1] → [N, F]
        self.potential_proj = nn.Linear(1, F, bias=True)

        # ── 标量初始融合 MLP ─────────────────────────────────────────────────
        # 输入：[s_atom ‖ s_pot] ∈ R^{2F}，输出：R^F
        self.scalar_init_mlp = nn.Sequential(
            nn.Linear(2 * F, F),
            nn.SiLU(),
            nn.Linear(F, F),
        )

        # ── MM 电场矢量投影（无 bias，保持等变性）────────────────────────────
        # E_i: [N, 3] 视为 1 通道矢量，投影为 F 通道
        # 实现：对各 channel 维度做线性映射，等价于 Linear(1→F, bias=False)
        # 存储约定：节点矢量 shape = [N, 3, F]
        # E_i[:, :, None] → [N, 3, 1]，乘以权重 [1, F] → [N, 3, F]
        self.vector_init_proj = nn.Linear(1, F, bias=False)

        # ── PaiNN 消息传递层（双向 QM-QM）───────────────────────────────────
        self.interactions = nn.ModuleList([
            layer.PaiNNInteraction(F, distance_embedding_size, cutoff)
            for _ in range(num_interactions)
        ])

        # ── PaiNN 更新层（标量-矢量交叉更新）───────────────────────────────
        self.updates = nn.ModuleList([
            layer.PaiNNUpdate(F)
            for _ in range(num_interactions)
        ])

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        V_flat: torch.Tensor,   # [N_total, 1]  MM 电势（来自 MMPhysicsField）
        E_flat: torch.Tensor,   # [N_total, 3]  MM 电场矢量
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        执行 L 层 PaiNN 消息传递，收集每层输出的特征轨迹。

        Returns:
            scalar_traj: List[Tensor]  长度 L，每项 shape [N_total, F]
            vector_traj: List[Tensor]  长度 L，每项 shape [N_total, 3, F]
        """
        # ── 展平边与节点（去 padding，拼接批次）────────────────────────────
        # edges_displacement 在孤立体系中全为零，calc_distance 内部乘以 cell 后仍为零
        edges_displacement = unpad_and_cat(
            input_dict["atom_edges_displacement"],
            input_dict["num_atom_edges"]
        )   # [E_total, 3]

        # 边索引的批次偏移：第 b 个样本的节点在展平后从 offset[b] 开始
        # cumsum([0, n0, n1, ...]) 的前 B 项即为各样本节点起始偏移
        edge_offset = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=input_dict["num_nodes"].device, dtype=torch.long),
                input_dict["num_nodes"][:-1],
            ]),
            dim=0,
        )   # [B]
        # 扩展为 [B, 1, 1] 以便与 atom_edges [B, E_max, 2] 广播相加
        edge_offset = edge_offset[:, None, None]                   # [B, 1, 1]
        edges = input_dict["atom_edges"] + edge_offset             # [B, E_max, 2]
        edges = unpad_and_cat(edges, input_dict["num_atom_edges"]) # [E_total, 2]

        # ── 节点坐标（展平）────────────────────────────────────────────────
        atom_xyz = unpad_and_cat(
            input_dict["atom_xyz"], input_dict["num_nodes"]
        )   # [N_total, 3]

        # ── 计算边的欧氏距离（孤立体系，displacement 全零）──────────────────
        # diff = atom_xyz[edges[:,1]] - atom_xyz[edges[:,0]]  [E_total, 3]
        # dist = ||diff||                                     [E_total, 1]
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )   # [E_total, 1], [E_total, 3]

        # ── sinc RBF 展开距离 ─────────────────────────────────────────────
        # 将标量距离展开为 distance_embedding_size 维的径向基函数特征
        # edge_state[e, k] = sin(k·π·d_e / r_cut) · π·k / r_cut
        edge_state = sinc_expansion(
            edges_distance,
            [(self.distance_embedding_size, self.cutoff)]
        )   # [E_total, distance_embedding_size]

        # ── 初始化节点标量特征 s_i^(0) ─────────────────────────────────────
        # 原子序数嵌入
        nodes_idx = unpad_and_cat(
            input_dict["nodes"], input_dict["num_nodes"]
        )   # [N_total]
        s_atom = self.atom_embedding(nodes_idx)             # [N_total, F]

        # MM 电势投影（连续物理量 → F 维）
        s_pot = self.potential_proj(V_flat)                 # [N_total, F]

        # 融合：[s_atom ‖ s_pot] → MLP → s^(0)
        # Shape: [N_total, 2F] → [N_total, F]
        nodes_scalar = self.scalar_init_mlp(
            torch.cat([s_atom, s_pot], dim=-1)
        )   # [N_total, F]

        # ── 初始化节点矢量特征 v_i^(0) ─────────────────────────────────────
        # E_flat: [N_total, 3]，视为 1 通道 3D 矢量
        # E_flat[:, :, None]: [N_total, 3, 1]
        # vector_init_proj: Linear(1→F, bias=False)，权重 [F, 1]
        # 结果：[N_total, 3, F]，每个 channel 是 E_i 的一个方向性投影
        # 物理含义：将 MM 电场的方向信息（各向异性环境感知）展开到 F 个等变通道
        # 严禁加 bias：bias 是各向同性常数，加入后会破坏矢量旋转等变性
        nodes_vector = self.vector_init_proj(
            E_flat[:, :, None]
        )   # [N_total, 3, F]

        # ── L 层 PaiNN 消息传递 + 更新，收集轨迹 ────────────────────────────
        scalar_traj: List[torch.Tensor] = []
        vector_traj: List[torch.Tensor] = []

        for interact_layer, update_layer in zip(self.interactions, self.updates):
            # Interaction: 原子间消息聚合，更新 (scalar, vector)
            nodes_scalar, nodes_vector = interact_layer(
                nodes_scalar,    # [N_total, F]
                nodes_vector,    # [N_total, 3, F]
                edge_state,      # [E_total, D_rbf]
                edges_diff,      # [E_total, 3]
                edges_distance,  # [E_total, 1]
                edges,           # [E_total, 2]
            )
            # Update: 节点内部标量-矢量交叉更新
            nodes_scalar, nodes_vector = update_layer(
                nodes_scalar, nodes_vector
            )
            scalar_traj.append(nodes_scalar)
            vector_traj.append(nodes_vector)

        # scalar_traj: L × [N_total, F]
        # vector_traj: L × [N_total, 3, F]
        return scalar_traj, vector_traj


# ==============================================================================
# §4  JK-Net 等变特征聚合
# ==============================================================================

class JKNetAggregation(nn.Module):
    """
    Jumping Knowledge Network 等变层级注意力聚合（§4）。

    收集 PaiNN 各层输出的特征轨迹，通过基于节点级注意力的加权求和，
    将多尺度特征压缩为单一的终态特征对 (S_JK, V_JK)。

    设计要点：
      ① 注意力权重的计算**仅依赖纯量特征**（旋转不变量），
         从而严格保证 SE(3) 等变性。
      ② 采用**加权求和（帽和）**而非特征拼接，
         因各层通道数均为 F，加权后维度自然保持 F，无需额外投影。
      ③ 注意力 MLP 输出 1 维 logit（每层一个），在层维度做 Softmax 归一化。

    数学表达（§4.1 & §4.2）：
        e_i^(l) = MLP_attn(s_i^(l))       ∈ R
        a_i^(l) = softmax_l({e_i^(k)})    ∈ R  (各层之和为 1)
        S_JK,i  = Σ_l  a_i^(l) · s_i^(l) ∈ R^F
        V_JK,i  = Σ_l  a_i^(l) · v_i^(l) ∈ R^{3×F}

    Args:
        num_layers  (int): PaiNN 层数 L（即特征轨迹长度）
        hidden_size (int): 隐藏层通道数基数 F
    """

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        F = hidden_size

        # 注意力评分 MLP：s_i^(l) [F] → logit [1]
        # 共享一个 MLP 处理所有层（参数量与层数无关），避免参数爆炸
        self.attn_mlp = nn.Sequential(
            nn.Linear(F, F // 2),
            nn.SiLU(),
            nn.Linear(F // 2, 1),
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
        L        = self.num_layers
        N_total  = scalar_traj[0].shape[0]
        device   = scalar_traj[0].device
        dtype    = scalar_traj[0].dtype

        # ── Step 4.1：计算各层注意力 logits ──────────────────────────────────
        # 将 L 个 [N, F] 堆叠为 [N, L, F]，批量通过 attn_mlp → [N, L, 1]
        # stack 在 dim=1 创建新的层维度
        s_stack = torch.stack(scalar_traj, dim=1)   # [N_total, L, F]

        # attn_mlp 输入 [N*L, F]，输出 [N*L, 1]，reshape 为 [N, L]
        # N*L → reshape → [N, L, 1] → squeeze → [N, L]
        logits = self.attn_mlp(
            s_stack.view(N_total * L, -1)
        ).view(N_total, L)                           # [N_total, L]

        # Softmax 在层维度 (dim=1) 归一化，确保各层权重之和为 1
        attn_weights = torch.softmax(logits, dim=1) # [N_total, L]

        # ── Step 4.2：加权求和得到终态特征 ──────────────────────────────────
        # 标量：a_i^(l) · s_i^(l)  → 求和
        # attn_weights[:, :, None]: [N, L, 1]  与 s_stack [N, L, F] 广播
        S_JK = (attn_weights[:, :, None] * s_stack).sum(dim=1)  # [N_total, F]

        # 矢量：a_i^(l) · v_i^(l)  → 求和
        # v_stack: [N_total, L, 3, F]
        # attn_weights[:, :, None, None]: [N, L, 1, 1]
        v_stack = torch.stack(vector_traj, dim=1)   # [N_total, L, 3, F]
        V_JK = (attn_weights[:, :, None, None] * v_stack).sum(dim=1)  # [N_total, 3, F]

        return S_JK, V_JK


# ==============================================================================
# §5  信息瓶颈读出网络（原子 → 探针）
# ==============================================================================

class InfoBottleneckReadout(nn.Module):
    """
    信息瓶颈读出网络：从终态原子特征 (S_JK, V_JK) 单跳预测探针密度（§5）。

    本模块仅使用聚合后的终态特征进行单次（single-hop）预测，
    不做多步扩散，从而充当信息瓶颈，迫使编码器在 JK 特征中压缩所有有用信息。

    流程（严格对应蓝图 §5.1 & §5.2）：

    Step 5.1 — 提取空间几何不变量：
        e_pi  = RBF(||r_p - r_i||)             ∈ R^{D_rbf}   距离特征
        q_pi  = <r̂_pi, V_JK,i>                ∈ R^F         方向性查询内积
        n_i   = ||V_JK,i||_2                   ∈ R^F         矢量模长

    Step 5.2 — 特征融合与消息生成：
        h_pi  = [e_pi ‖ q_pi ‖ n_i ‖ S_JK,i]  ∈ R^{D_rbf+3F}
        m_pi  = MLP_readout(h_pi)              ∈ R          原子 i 对探针 p 的贡献

    Step 5.3 — 截断加权聚合：
        ρ_p   = Σ_{i∈N(p)}  m_pi · C(d_pi)    标量密度输出

        其中 C(d) 为 Behler-Parinello cosine 截断函数（DimeNet 风格），
        确保贡献在截断边界处平滑衰减至 0。

    Args:
        hidden_size  (int):   隐藏层通道数基数 F
        D_rbf        (int):   RBF 展开维度
        cutoff       (float): 探针-原子截断半径（与 QM-QM 图截断共用）
        distance_embedding_size (int): sinc RBF 的展开数 n
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
        F = hidden_size

        # ── 消息 MLP：h_pi [D_rbf + 3F] → m_pi [1] ────────────────────────
        # 输入拼接维度：D_rbf (距离) + F (内积) + F (模长) + F (标量)
        in_dim = D_rbf + 3 * F
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, F),
            nn.SiLU(),
            nn.Linear(F, F // 2),
            nn.SiLU(),
            nn.Linear(F // 2, 1),
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        S_JK: torch.Tensor,   # [N_total, F]    终态标量特征
        V_JK: torch.Tensor,   # [N_total, 3, F] 终态矢量特征
    ) -> torch.Tensor:
        """
        Returns:
            probe_output: [B, P_max]  每个批次样本的探针密度预测（含 padding）
        """
        device = S_JK.device
        dtype  = S_JK.dtype

        # ── 展平探针坐标与节点坐标 ────────────────────────────────────────
        atom_xyz = unpad_and_cat(
            input_dict["atom_xyz"], input_dict["num_nodes"]
        )   # [N_total, 3]

        probe_xyz = unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )   # [P_total, 3]

        # ── 构建探针边偏移 ────────────────────────────────────────────────
        # probe_edges[b, e, :] = [atom_idx_local, probe_idx_local]
        # 需要分别加上原子偏移和探针偏移才能在展平维度中正确索引

        # 原子全局偏移（各样本节点起始位置）
        atom_offset = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                input_dict["num_nodes"][:-1],
            ]),
            dim=0,
        )   # [B]

        # 探针全局偏移（各样本探针起始位置）
        probe_offset = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                input_dict["num_probes"][:-1],
            ]),
            dim=0,
        )   # [B]

        # 合并偏移为 [B, 1, 2]（第 0 列加 atom_offset，第 1 列加 probe_offset）
        combined_offset = torch.stack(
            [atom_offset, probe_offset], dim=1
        )[:, None, :]   # [B, 1, 2]

        probe_edges = input_dict["probe_edges"] + combined_offset   # [B, E_max, 2]
        probe_edges = unpad_and_cat(
            probe_edges, input_dict["num_probe_edges"]
        )   # [E_probe_total, 2]  col0=atom_idx, col1=probe_idx

        # ── 计算探针-原子距离与方向向量 ──────────────────────────────────
        # diff_pi = r_p - r_i    [E_probe_total, 3]
        # dist_pi = ||diff_pi||  [E_probe_total, 1]
        probe_edges_displacement = unpad_and_cat(
            input_dict["probe_edges_displacement"],
            input_dict["num_probe_edges"]
        )   # [E_probe_total, 3]，孤立体系全零

        dist_pi, diff_pi = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )   # [E_probe_total, 1], [E_probe_total, 3]

        # ── Step 5.1(a)：RBF 距离特征 e_pi ─────────────────────────────────
        # sinc 展开：e_pi[k] = sin(k·π·d / r_cut) · πk/r_cut
        e_pi = sinc_expansion(
            dist_pi,
            [(self.distance_embedding_size, self.cutoff)]
        )   # [E_probe_total, D_rbf]

        # ── Step 5.1(b)：方向性查询内积 q_pi ─────────────────────────────
        # 单位方向向量 r̂_pi = diff_pi / ||diff_pi||  [E_probe_total, 3]
        # 注意：diff_pi = probe - atom，方向为"从原子指向探针"
        r_hat_pi = diff_pi / (
            torch.linalg.norm(diff_pi, dim=-1, keepdim=True) + 1e-8
        )   # [E_probe_total, 3]

        # 获取每条边对应的原子索引（边的发送端）
        atom_idx = probe_edges[:, 0]   # [E_probe_total]

        # V_JK,i[atom_idx]: [E_probe_total, 3, F]
        # r̂_pi:              [E_probe_total, 3]
        #
        # 方向性查询内积：q_pi = <r̂_pi, V_JK,i>
        #   对 3D 空间维（dim=1）做点积，得到旋转不变标量特征
        #   公式：q_pi[f] = Σ_{xyz}  r̂_pi[xyz] · V_JK,i[xyz, f]
        #
        # 维度追踪：
        #   V_JK_e:   [E, 3, F]
        #   r_hat_pi: [E, 3, 1]  (unsqueeze 对齐 F 维)
        #   inner:    [E, 3, F] * [E, 3, 1] → sum(dim=1) → [E, F]
        V_JK_e = V_JK[atom_idx]                                      # [E, 3, F]
        q_pi = (V_JK_e * r_hat_pi[:, :, None]).sum(dim=1)            # [E, F]

        # ── Step 5.1(c)：矢量模长 n_i ──────────────────────────────────────
        # n_i[f] = ||V_JK,i[:, f]||_2  沿 3D 空间维求范数
        # n_i[atom_idx]: [E, F]，每条边取发送原子的模长
        n_i = torch.linalg.norm(V_JK, dim=1)                        # [N_total, F]
        n_e = n_i[atom_idx]                                          # [E, F]

        # ── Step 5.2：消息拼接与 MLP ─────────────────────────────────────
        # h_pi = [e_pi ‖ q_pi ‖ n_e ‖ S_JK,i]
        # 维度：[D_rbf + F + F + F] = [D_rbf + 3F]
        S_JK_e = S_JK[atom_idx]                                      # [E, F]
        h_pi = torch.cat([e_pi, q_pi, n_e, S_JK_e], dim=-1)         # [E, D_rbf+3F]

        # 消息值 m_pi：[E, 1] → squeeze → [E]
        m_pi = self.msg_mlp(h_pi).squeeze(-1)                        # [E]

        # ── Step 5.3：截断函数加权聚合 ──────────────────────────────────
        # C(d_pi)：cosine 截断函数，在 d → r_cut 时平滑衰减至 0
        cutoff_weight = cosine_cutoff(dist_pi, self.cutoff).squeeze(-1)  # [E]
        m_weighted = m_pi * cutoff_weight                            # [E]

        # 将所有边的消息按探针索引聚合（scatter add）
        probe_idx     = probe_edges[:, 1]                            # [E]
        P_total       = probe_xyz.shape[0]
        rho_flat = torch.zeros(P_total, device=device, dtype=dtype)  # [P_total]
        rho_flat.index_add_(0, probe_idx, m_weighted)                # [P_total]

        # ── 重新 pad & stack 为批次格式 [B, P_max] ─────────────────────
        probe_output = pad_and_stack(
            torch.split(
                rho_flat,
                input_dict["num_probes"].tolist(),
                dim=0,
            )
        )   # [B, P_max]

        return probe_output


# ==============================================================================
# 顶层容器：PaiNN 量子感知密度模型
# ==============================================================================

class PaiNNQMEncoder(nn.Module):
    """
    PaiNN 量子感知在编码器顶层容器（对应设计蓝图第一阶段）。

    整合全部五个子模块，完成从 QM/MM 输入到探针密度预测的完整前向传播，
    并通过 `get_jk_features()` 将 JK 终态特征作为 API 暴露给下游 SBDD 模型。

    子模块：
        mm_physics    — §2  MMPhysicsField
        atom_encoder  — §3  PaiNNAtomEncoder
        jk_aggregator — §4  JKNetAggregation
        readout       — §5  InfoBottleneckReadout

    Args:
        num_interactions       (int):   PaiNN 消息传递层数 L
        hidden_size            (int):   隐藏层通道数基数 F（蓝图建议 128）
        cutoff                 (float): QM-QM / 探针-原子截断半径（Å）
        distance_embedding_size(int):   sinc RBF 展开维度（默认 = hidden_size // 2）
        D_rbf                  (int):   读出网络 RBF 维度（默认 = hidden_size // 2）
        mm_eps                 (float): 物理场计算数值稳定小量
    """

    def __init__(
        self,
        num_interactions:       int   = 3,
        hidden_size:            int   = 128,
        cutoff:                 float = 4.0,
        distance_embedding_size: Optional[int]   = None,
        D_rbf:                  Optional[int]   = None,
        mm_eps:                 float = 1e-8,
    ):
        super().__init__()
        self.num_interactions = num_interactions
        self.hidden_size      = hidden_size
        self.cutoff           = cutoff

        F = hidden_size
        # 默认值：RBF 维度 = F // 2，足以精细描述 0~cutoff 的距离分布
        _dist_emb = distance_embedding_size if distance_embedding_size is not None else F // 2
        _D_rbf    = D_rbf if D_rbf is not None else F // 2

        self.distance_embedding_size = _dist_emb
        self.D_rbf = _D_rbf

        # ── §2：MM 静电物理场 ────────────────────────────────────────────
        self.mm_physics = MMPhysicsField(eps=mm_eps)

        # ── §3：PaiNN 原子编码器 ─────────────────────────────────────────
        self.atom_encoder = PaiNNAtomEncoder(
            num_interactions        = num_interactions,
            hidden_size             = F,
            cutoff                  = cutoff,
            distance_embedding_size = _dist_emb,
        )

        # ── §4：JK-Net 聚合 ─────────────────────────────────────────────
        self.jk_aggregator = JKNetAggregation(
            num_layers  = num_interactions,
            hidden_size = F,
        )

        # ── §5：信息瓶颈读出 ─────────────────────────────────────────────
        self.readout = InfoBottleneckReadout(
            hidden_size             = F,
            D_rbf                   = _D_rbf,
            cutoff                  = cutoff,
            distance_embedding_size = _dist_emb,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 内部辅助：运行 §2–§4，返回 JK 终态特征
    # ──────────────────────────────────────────────────────────────────────

    def _encode_atoms(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 §2（物理场）+ §3（PaiNN 编码）+ §4（JK 聚合），
        返回终态特征 (S_JK, V_JK)。

        这是暴露给下游 SBDD 模型的"截断口"：
        下游可直接调用此方法，跳过 §5 读出网络，获取节点级条件特征。

        Returns:
            S_JK: [N_total, F]    终态标量特征
            V_JK: [N_total, 3, F] 终态等变矢量特征
        """
        # §2：计算 QM 原子处的 MM 静电势与电场
        atom_xyz_flat = unpad_and_cat(
            input_dict["atom_xyz"], input_dict["num_nodes"]
        )

        V_flat, E_flat = self.mm_physics(
            qm_positions  = atom_xyz_flat,
            mm_positions  = input_dict["mm_positions"],   # [B, M_max, 3]
            mm_charges    = input_dict["mm_charges"],     # [B, M_max]
            num_nodes     = input_dict["num_nodes"],      # [B]
            num_mm_atoms  = input_dict["num_mm_atoms"],   # [B]
        )

        # §3：PaiNN L 层消息传递，收集特征轨迹
        scalar_traj, vector_traj = self.atom_encoder(
            input_dict, V_flat, E_flat
        )

        # §4：JK-Net 注意力聚合
        S_JK, V_JK = self.jk_aggregator(scalar_traj, vector_traj)

        return S_JK, V_JK

    # ──────────────────────────────────────────────────────────────────────
    # 公开 API：暴露 JK 特征给下游模型
    # ──────────────────────────────────────────────────────────────────────

    def get_jk_features(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【下游 SBDD 接口】仅运行 §2–§4，返回节点级 JK 终态条件特征。

        下游扩散模型等 SBDD 方法可调用此接口，按样本切割 S_JK / V_JK
        作为原子级 conditioning features，无需触发探针读出层。

        Returns:
            S_JK: [N_total, F]    终态标量特征（旋转不变）
            V_JK: [N_total, 3, F] 终态等变矢量特征（SE(3) 等变）
        """
        return self._encode_atoms(input_dict)

    # ──────────────────────────────────────────────────────────────────────
    # 完整前向传播（§2–§5）：用于预训练密度预测任务
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
        # §2–§4：原子编码，得到 JK 终态特征
        S_JK, V_JK = self._encode_atoms(input_dict)

        # §5：信息瓶颈读出，预测探针密度
        probe_output = self.readout(input_dict, S_JK, V_JK)

        return probe_output

    # ──────────────────────────────────────────────────────────────────────
    # 分步接口：供 Trainer._test_step 缓存 atom_repr 复用
    # ──────────────────────────────────────────────────────────────────────

    def atom_model(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅执行原子编码（§2–§4），返回 (S_JK, V_JK)。
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
        仅执行探针读出（§5），复用已缓存的原子特征。
        命名与旧 PainnDensityModel.probe_model 对齐，方便 Trainer 复用。
        """
        return self.readout(input_dict, S_JK, V_JK)
