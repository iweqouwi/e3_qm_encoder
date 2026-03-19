"""
QM/MM 量子环境编码器 — 预训练入口脚本（无 Hydra，传统实例化）

设计目标：
  - 直接解析 YAML 配置，不依赖 Hydra/OmegaConf 魔法注入
  - 单机多卡 DDP（torch.multiprocessing.spawn）
  - 所有组件显式实例化，便于调试和修改

启动方式（4×RTX 3090 单机）：
  python src/train_qmmm.py --config configs/train_qmmm_config.yaml

断点续训：
  python src/train_qmmm.py --config configs/train_qmmm_config.yaml \
         --checkpoint results/qmmm_encoder/checkpoint.pt
"""

import os
import sys
import math
import argparse
import yaml
from functools import partial

# 确保项目根目录在 Python 路径中，以便 src.* 包导入正常工作
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# ── 项目内部模块导入 ─────────────────────────────────────────────────────────
from src.charge3net.data.graph_construction import GraphConstructor
from src.charge3net.data.dataset import DensityDatamodule
from src.charge3net.models.e3 import QMEnvironmentEncoder
from src.charge3net.models.loss import LogMSELoss
from src.trainer import Trainer


# =============================================================================
# 单进程训练函数（每张 GPU 独立运行一份）
# =============================================================================

def train_worker(rank: int, cfg: dict, env: dict):
    """
    DDP 子进程的实际训练逻辑。由 mp.spawn 自动注入 rank 参数。

    Args:
        rank (int): 当前进程在本机的 GPU 编号（0 ~ nprocs-1）
        cfg  (dict): 从 YAML 解析的完整配置字典
        env  (dict): DDP 通信环境参数（master_addr, master_port, world_size 等）
    """

    # ── 1. 固定随机种子 ──────────────────────────────────────────────────────
    # 每张 GPU 使用不同种子保证数据打乱的多样性，但基于统一的 base seed
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)

    # ── 2. 计算全局进程编号 (Global Rank) ────────────────────────────────────
    # 单机训练：group_rank = 0，global_rank = rank
    # 多机训练：group_rank = 当前节点编号，global_rank 跨所有节点唯一
    global_rank = env["group_rank"] * cfg["nprocs"] + rank

    # ── 3. 初始化 NCCL 分布式通信组 ─────────────────────────────────────────
    init_process_group(
        backend="nccl",
        init_method="env://",
        rank=global_rank,
        world_size=env["world_size"],
    )
    torch.cuda.set_device(rank)   # 强制当前进程仅使用被分配的那张显卡

    print(f"[Rank {global_rank}/{env['world_size']}] 初始化完成，使用 GPU cuda:{rank}")

    # =========================================================================
    # 4. 实例化图构建器（GraphConstructor）
    # =========================================================================
    # GraphConstructor 是一个 callable，__call__(data_dict) 将 HDF5 原始数据
    # 转换为模型所需的图字典（节点、边、探针坐标等）。
    # DensityDatamodule 要求传入一个"工厂函数"，即接受 num_probes 参数后
    # 返回 GraphConstructor 实例的函数，因此这里用 partial 绑定 cutoff。
    graph_constructor_factory = partial(
        GraphConstructor,
        cutoff=cfg["graph"]["cutoff"],
    )
    # partial 绑定后：graph_constructor_factory(num_probes=500) 等价于
    # GraphConstructor(cutoff=4.0, num_probes=500)

    # =========================================================================
    # 5. 实例化数据模块（DensityDatamodule）
    # =========================================================================
    # DensityDatamodule 负责：
    #   - 读取 HDF5 数据（支持目录扫描所有分片 .h5 文件）
    #   - 按 val_frac / split_file 划分 train/val/test
    #   - 为三个分区分别创建 GraphConstructor（探针数各不相同）
    #   - 提供 train_dataloader() / val_dataloader() / test_dataloader()
    d = cfg["data"]
    datamodule = DensityDatamodule(
        data_root=d["data_root"],
        graph_constructor=graph_constructor_factory,
        train_probes=d["train_probes"],
        val_probes=d["val_probes"],
        test_probes=d.get("test_probes", d["val_probes"]),
        batch_size=d["batch_size"],
        train_workers=d["train_workers"],
        val_workers=d["val_workers"],
        pin_memory=d.get("pin_memory", False),
        val_frac=d["val_frac"],
        split_file=d.get("split_file", None),
    )

    # =========================================================================
    # 6. 实例化模型（QMEnvironmentEncoder）
    # =========================================================================
    # QMEnvironmentEncoder 整合了四个子模块：
    #   ① MMPhysicsFeatureComputer  — 库仑场注入 (Task 1)
    #   ② QMAtomEncoder             — E3 等变消息传递编码 (Task 2 前置)
    #   ③ EquivariantFusionBottleneck — JK-Attention CG 融合 (Task 2)
    #   ④ EnhancedProbeReadoutModel — 单跳探针读出 (Task 3)
    m = cfg["model"]
    model = QMEnvironmentEncoder(
        num_interactions=m["num_interactions"],  # 消息传递层数 K
        mul=m["mul"],                            # irrep 通道倍数
        lmax=m["lmax"],                          # 最大角量子数
        # num_species: 由 QMAtomEncoder 内部通过 len(ase.data.atomic_numbers) 自动确定
        cutoff=cfg["graph"]["cutoff"],           # QM-QM 消息截断半径 & 探针读出截断半径 (Å)
        D_s=m["D_s"],                            # S_final 标量维度
        D_v=m["D_v"],                            # V_final 矢量通道数
        D_rbf=m["D_rbf"],                        # 径向基展开维度
        # probe_cutoff: 与 cutoff 共用，无独立参数
    )
    
    # =========================================================================
    # 【新增防御】解决均值漂移困境，初始化网络最后一层的偏置
    # =========================================================================
    def init_bias_for_log_density(module):
        # 匹配探针读出模块中的最后一层线性变换，将其 bias 设置为物理密度对数的经验均值
        if isinstance(module, torch.nn.Linear) and module.out_features == 1:
            torch.nn.init.constant_(module.bias, -11.5)
            
    model.apply(init_bias_for_log_density)
    if global_rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量：{n_params:,} ({n_params/1e6:.1f}M)")

    # =========================================================================
    # 7. 实例化损失函数（LogMSELoss）
    # =========================================================================
    # L = MSE(log(|ρ_pred| + ε), log(|ρ_true| + ε))
    # 对数空间中的 MSE 强制模型同等关注 NCI 弱相互作用区域和化学键强区域
    l = cfg["loss"]
    criterion = LogMSELoss(
        eps=l["eps"],
        mse_weight=l["mse_weight"],
    )

    # =========================================================================
    # 8. 实例化优化器（AdamW）
    # =========================================================================
    o = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=o["lr"],
        weight_decay=o.get("weight_decay", 0.0),
        betas=tuple(o.get("betas", [0.9, 0.999])),
        eps=o.get("eps", 1e-8),
    )

    # =========================================================================
    # 9. 实例化学习率调度器（指数衰减 ExponentialLR）
    # =========================================================================
    # 公式：lr_t = lr_0 × gamma^t，gamma = 0.1^(1/beta)
    # 物理意义：每经过 beta 步，学习率衰减到当前值的 10%
    # 与原 QM9 训练参数一致（beta=6000）
    beta = cfg["scheduler"]["beta"]
    gamma = math.exp(math.log(0.1) / beta)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # =========================================================================
    # 10. 组装训练控制器（Trainer）并启动训练
    # =========================================================================
    # Trainer 封装了：
    #   - DDP 模型包装
    #   - 训练循环（_train_epoch）：forward → criterion → backward → step
    #   - 验证循环（_valid_epoch）：NMAPE 指标计算
    #   - 最优 checkpoint 保存（val NMAPE 最低时保存）
    #   - TensorBoard 日志写入
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        log_dir=cfg["log_dir"],
        gpu_id=rank,
        global_rank=global_rank,
        load_checkpoint_path=cfg.get("checkpoint_path", None),
        log_steps=cfg.get("log_steps", 50),
    )

    # 获取 DataLoader
    # train_dataloader() 使用 DistributedSampler，自动按 global_rank 切分数据
    # val_dataloader() 无 DistributedSampler，所有 GPU 独立验证（Trainer 仅在 rank=0 记录）
    train_dl = datamodule.train_dataloader()
    val_dl   = datamodule.val_dataloader()

    if global_rank == 0:
        print(f"训练集大小：{len(datamodule.train_set)} 样本")
        print(f"验证集大小：{len(datamodule.val_set)} 样本")
        print(f"目标训练步数：{cfg['steps']:,}")
        print("开始训练...")

    trainer.fit(
        train_dl=train_dl,
        valid_dl=val_dl,
        steps=cfg["steps"],
    )

    # ── 11. 清理 DDP 资源 ─────────────────────────────────────────────────────
    destroy_process_group()


# =============================================================================
# 主进程入口：解析配置 → 设置 DDP 环境变量 → spawn 子进程
# =============================================================================

def main():
    # ── 命令行参数解析 ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="QMEnvironmentEncoder 预训练（无 Hydra）"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/train_qmmm_config.yaml",
        help="YAML 配置文件路径（默认：configs/train_qmmm_config.yaml）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="断点续训：覆盖 YAML 中的 checkpoint_path 字段",
    )
    args = parser.parse_args()

    # ── 读取 YAML 配置 ────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行 --checkpoint 优先级高于 YAML 中的配置
    if args.checkpoint is not None:
        cfg["checkpoint_path"] = args.checkpoint

    # ── 构建 DDP 通信环境字典 ─────────────────────────────────────────────────
    env = {
        "master_addr": "localhost",               # 单机训练固定 localhost
        "master_port": str(cfg.get("master_port", 29500)),
        "world_size":  cfg["nnodes"] * cfg["nprocs"],   # 总 GPU 数
        "group_rank":  int(os.environ.get("SLURM_NODEID", "0")),  # 单机为 0
    }

    # 将通信参数写入环境变量（NCCL 后端通过环境变量发现 master 节点）
    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]
    os.environ["WORLD_SIZE"]  = str(env["world_size"])

    print(f"配置文件：{args.config}")
    print(f"数据路径：{cfg['data']['data_root']}")
    print(f"GPU 数量：{cfg['nprocs']} × 本机 (world_size={env['world_size']})")
    print(f"模型：QMEnvironmentEncoder  "
          f"(K={cfg['model']['num_interactions']}, "
          f"mul={cfg['model']['mul']}, "
          f"lmax={cfg['model']['lmax']})")

    # ── 启动多进程训练 ────────────────────────────────────────────────────────
    # mp.spawn 在本机启动 nprocs 个子进程，每个进程运行 train_worker(rank, cfg, env)
    # rank 由 spawn 自动注入（0, 1, 2, 3 对应四张 GPU）
    mp.spawn(
        fn=train_worker,
        args=(cfg, env),
        nprocs=cfg["nprocs"],
        join=True,   # 等待所有子进程结束后 main() 才返回
    )

    print("训练完成。")


if __name__ == "__main__":
    main()
