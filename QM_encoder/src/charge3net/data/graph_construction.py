# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from scipy.spatial import KDTree

import torch
import numpy as np


class GraphConstructor(object):
    """
    图构建器：将 QM/MM 体系数据转换为图结构（节点和边）。
    - QM 区域原子作为节点
    - QM-QM 原子间边：基于截断半径
    - 探针-QM 边：每个探针点连接其截断半径内的 QM 原子
    - 无周期性边界条件（非晶体体系）
    """

    def __init__(self, cutoff, num_probes=None):
        """
        - cutoff (float): 截断半径（单位 Å）
        - num_probes (int, 可选): 训练时随机采样的探针数量；None 表示使用全部探针
        """
        super().__init__()
        self.cutoff = cutoff
        self.num_probes = num_probes
        self.default_type = torch.get_default_dtype()

    def __call__(self, data_dict):
        """
        将 HDF5 读入的 QM/MM 数据字典转换为模型所需的图字典。

        data_dict 字段：
            qm_positions  [N, 3] float32  QM 区域原子坐标
            qm_numbers    [N]    int16    QM 区域原子序数
            mm_positions  [M, 3] float32  MM 区域原子坐标
            mm_charges    [M]    float32  MM 区域原子偏电荷
            probe_xyz     [P, 3] float32  预设探针坐标
            probe_target  [P]    float32  探针处的电子密度目标值
        """
        qm_positions = data_dict["qm_positions"].astype(np.float32)   # [N, 3]
        qm_numbers   = data_dict["qm_numbers"].astype(np.int64)        # [N]
        mm_positions = data_dict["mm_positions"].astype(np.float32)   # [M, 3]
        mm_charges   = data_dict["mm_charges"].astype(np.float32)     # [M]
        probe_xyz    = data_dict["probe_xyz"].astype(np.float32)      # [P, 3]
        probe_target = data_dict["probe_target"].astype(np.float32)   # [P]

        # 训练时按 num_probes 随机子采样探针
        if self.num_probes is not None:
            n = min(self.num_probes, len(probe_xyz))
            idx = np.random.choice(len(probe_xyz), size=n, replace=False)
            probe_xyz    = probe_xyz[idx]
            probe_target = probe_target[idx]

        # 构建 QM-QM 原子间边
        atom_edges = self._qm_to_graph(qm_positions)

        # 构建探针-QM 边
        probe_edges = self._probes_to_graph(qm_positions, probe_xyz)

        # 位移全部为零（无 PBC），cell 设为单位矩阵（模型中 displacement * cell，全零位移结果也为零）
        graph_dict = {
            "nodes":                    torch.tensor(qm_numbers),
            "atom_edges":               torch.tensor(atom_edges),
            "atom_edges_displacement":  torch.zeros(len(atom_edges), 3, dtype=self.default_type),
            "probe_edges":              torch.tensor(probe_edges),
            "probe_edges_displacement": torch.zeros(len(probe_edges), 3, dtype=self.default_type),
            "probe_target":             torch.tensor(probe_target, dtype=self.default_type),
            "num_nodes":                torch.tensor(len(qm_numbers)),
            "num_atom_edges":           torch.tensor(len(atom_edges)),
            "num_probes":               torch.tensor(len(probe_target)),
            "num_probe_edges":          torch.tensor(len(probe_edges)),
            "probe_xyz":                torch.tensor(probe_xyz, dtype=self.default_type),
            "atom_xyz":                 torch.tensor(qm_positions, dtype=self.default_type),
            "cell":                     torch.eye(3, dtype=self.default_type),
            "mm_positions":             torch.tensor(mm_positions, dtype=self.default_type),
            "mm_charges":               torch.tensor(mm_charges, dtype=self.default_type),
            # MMPhysicsFeatureComputer 需要知道每个样本有多少真实 MM 原子（而非 padding）
            "num_mm_atoms":             torch.tensor(len(mm_charges)),
        }
        return graph_dict

    def _qm_to_graph(self, qm_positions):
        """
        用 KDTree 在截断半径内搜索 QM-QM 原子对，构建双向边。
        返回 atom_edges [E, 2]，其中每列为 (src, dst)。
        """
        if len(qm_positions) == 0:
            return np.zeros((0, 2), dtype=np.int64)

        tree = KDTree(qm_positions)
        # query_pairs 返回 i < j 的无向对
        pairs = np.array(list(tree.query_pairs(r=self.cutoff)), dtype=np.int64)

        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64)

        # 双向边
        atom_edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        return atom_edges

    def _probes_to_graph(self, qm_positions, probe_xyz):
        """
        用 KDTree 为每个探针找到截断半径内的 QM 原子，构建 (atom_idx, probe_idx) 边。
        返回 probe_edges [E, 2]。
        """
        if len(probe_xyz) == 0 or len(qm_positions) == 0:
            return np.zeros((0, 2), dtype=np.int64)

        atom_tree  = KDTree(qm_positions)
        probe_tree = KDTree(probe_xyz)

        # 每个探针点的邻近原子列表
        query = probe_tree.query_ball_tree(atom_tree, r=self.cutoff)

        edges_per_probe = [len(q) for q in query]
        if sum(edges_per_probe) == 0:
            return np.zeros((0, 2), dtype=np.int64)

        dest_idx = np.concatenate(
            [[i] * n for i, n in enumerate(edges_per_probe)]
        ).astype(np.int64)
        src_idx = np.concatenate(query).astype(np.int64)

        probe_edges = np.stack((src_idx, dest_idx), axis=1)
        return probe_edges


def _sort_by_rows(arr):
    assert len(arr.shape) == 2, "Only 2D arrays"
    return np.array(sorted([tuple(x) for x in arr.tolist()]))
