"""
QM/MM 量子环境编码器 — 预训练入口脚本（无 Hydra，传统实例化）

设计目标：
  - 直接解析 YAML 配置，不依赖 Hydra/OmegaConf 魔法注入
  - 单机多卡 DDP（torch.multiprocessing.spawn）
  - 支持两套模型后端，通过 YAML model.backend 字段切换：
      "painn" → PaiNNQMEncoder   （densitymodel.py，轻量高效，推荐预训练）
      "e3nn"  → QMEnvironmentEncoder（e3.py，高精度 E3 等变架构）
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
from src.charge3net.models.densitymodel import PaiNNQMEncoder
from src.charge3net.models.e3 import QMEnvironmentEncoder
from src.charge3net.models.loss import LogMSELoss
from src.trainer import Trainer


# =============================================================================
# 模型工厂：根据 backend 字段实例化对应模型
# =============================================================================

def build_model(cfg: dict, cutoff: float) -> torch.nn.Module:
    """
    根据 cfg["model"]["backend"] 实例化对应模型。

    Args:
        cfg    (dict):  完整配置字典
        cutoff (float): 来自 cfg["graph"]["cutoff"] 的截断半径

    Returns:
        torch.nn.Module: 已实例化的模型（尚未移至 GPU）

    Raises:
        ValueError: backend 字段不在 {"painn", "e3nn"} 中时抛出
    """
    m = cfg["model"]
    backend = m.get("backend", "painn").lower()

    if backend == "painn":
        # ── PaiNNQMEncoder（新架构，densitymodel.py §2–§5）──────────────────
        p = m["painn"]

        # distance_embedding_size / D_rbf 支持 null（由模型内部自动推导）
        dist_emb = p.get("distance_embedding_size") or None
        d_rbf    = p.get("D_rbf") or None

        model = PaiNNQMEncoder(
            num_interactions        = p["num_interactions"],
            hidden_size             = p["hidden_size"],
            cutoff                  = cutoff,
            distance_embedding_size = dist_emb,
            D_rbf                   = d_rbf,
            num_heads               = p.get("num_heads", 4),
            mm_eps                  = p.get("mm_eps", 1e-8),
        )

    elif backend == "e3nn":
        # ── QMEnvironmentEncoder（旧架构，e3.py E3 等变）─────────────────────
        e = m["e3nn"]
        model = QMEnvironmentEncoder(
            num_interactions = e["num_interactions"],
            mul              = e["mul"],
            lmax             = e["lmax"],
            cutoff           = cutoff,
            D_s              = e["D_s"],
            D_v              = e["D_v"],
            D_rbf            = e["D_rbf"],
        )
        # E3nn 同样需要偏置初始化, 批判使用这个，这个是根据训练数据统计得到的玄学值，能显著提升训练初期的稳定性和收敛速度
        _init_output_bias(model, -6.10)

    else:
        raise ValueError(
            f"未知的模型后端 backend='{backend}'，"
            f"请在 YAML 中将 model.backend 设为 'painn' 或 'e3nn'。"
        )

    return model


def _init_output_bias(model: torch.nn.Module, bias_value: float) -> None:
    """
    遍历模型，将所有 out_features==1 的 Linear 层的 bias 初始化为 bias_value。

    物理动机：模型输出的是对数密度 log(ρ)
    平均值 (Mean): 3.01e-3 a.u. ≈ 0.082 eV/Å³，ln(3.01e-3) ≈ -5.81
    中位数 (P50): 2.23e-3 a.u. ≈ 0.057 eV/Å³，ln(2.23e-3) ≈ -6.10，根据训练数据统计得到的玄学
    将最终输出层 bias 初始化为该值，可避免训练初期因输出偏离真实量级
    而产生的极大梯度，显著提升收敛速度和训练稳定性。

    Args:
        model      (nn.Module): 待初始化的模型
        bias_value (float):     bias 初始化目标值
    """
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Linear)
            and module.out_features == 1
            and module.bias is not None
        ):
            torch.nn.init.constant_(module.bias, bias_value)


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
    # 这是一个调试选项，开启后 PyTorch 在反向传播时会检查异常（如 NaN/Inf）并提供完整的堆栈跟踪，便于定位问题
    # torch.autograd.set_detect_anomaly(True)
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

    # =========================================================================
    # 5. 实例化数据模块（DensityDatamodule）
    # =========================================================================
    d = cfg["data"]
    datamodule = DensityDatamodule(
        data_root          = d["data_root"],
        graph_constructor  = graph_constructor_factory,
        train_probes       = d["train_probes"],
        val_probes         = d["val_probes"],
        test_probes        = d.get("test_probes", d["val_probes"]),
        batch_size         = d["batch_size"],
        train_workers      = d["train_workers"],
        val_workers        = d["val_workers"],
        pin_memory         = d.get("pin_memory", False),
        val_frac           = d["val_frac"],
        split_file         = d.get("split_file", None),
    )

    # =========================================================================
    # 6. 实例化模型（根据 backend 字段分发）
    # =========================================================================
    model = build_model(cfg, cutoff=cfg["graph"]["cutoff"])

    if global_rank == 0:
        backend = cfg["model"].get("backend", "painn").lower()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"模型后端：{backend.upper()}  |  "
            f"参数量：{n_params:,} ({n_params / 1e6:.2f}M)"
        )

    # =========================================================================
    # 7. 实例化损失函数（LogMSELoss）
    # =========================================================================
    # L = MSE(log(|ρ_pred| + ε), log(|ρ_true| + ε))
    # 对数空间中的 MSE 强制模型同等关注 NCI 弱相互作用区域和化学键强区域
    l = cfg["loss"]
    criterion = LogMSELoss(
        eps        = l["eps"],
        mse_weight = l["mse_weight"],
    )

    # =========================================================================
    # 8. 实例化优化器（AdamW）
    # =========================================================================
    o = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = o["lr"],
        weight_decay = o.get("weight_decay", 0.0),
        betas        = tuple(o.get("betas", [0.9, 0.999])),
        eps          = o.get("eps", 1e-8),
    )

    # =========================================================================
    # 9. 实例化学习率调度器（ExponentialLR）
    # =========================================================================
    # 公式：lr_t = lr_0 × gamma^t，gamma = 0.1^(1/beta)
    # 物理意义：每经过 beta 步，学习率衰减到当前值的 10%
    beta  = cfg["scheduler"]["beta"]
    gamma = math.exp(math.log(0.1) / beta)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # =========================================================================
    # 10. 组装训练控制器（Trainer）并启动训练
    # =========================================================================
    trainer = Trainer(
        model               = model,
        optimizer           = optimizer,
        scheduler           = scheduler,
        criterion           = criterion,
        log_dir             = cfg["log_dir"],
        gpu_id              = rank,
        global_rank         = global_rank,
        load_checkpoint_path= cfg.get("checkpoint_path", None),
        log_steps           = cfg.get("log_steps", 50),
    )

    train_dl = datamodule.train_dataloader()
    val_dl   = datamodule.val_dataloader()

    if global_rank == 0:
        print(f"训练集大小：{len(datamodule.train_set)} 样本")
        print(f"验证集大小：{len(datamodule.val_set)} 样本")
        print(f"目标训练步数：{cfg['steps']:,}")
        print("开始训练...")

    trainer.fit(
        train_dl = train_dl,
        valid_dl = val_dl,
        steps    = cfg["steps"],
    )

    # ── 11. 清理 DDP 资源 ─────────────────────────────────────────────────────
    destroy_process_group()


# =============================================================================
# 主进程入口：解析配置 → 设置 DDP 环境变量 → spawn 子进程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QMEnvironmentEncoder 预训练（无 Hydra）"
    )
    parser.add_argument(
        "--config", "-c",
        type    = str,
        default = "configs/train_painn_config.yaml",
        help    = "YAML 配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type    = str,
        default = None,
        help    = "断点续训：覆盖 YAML 中的 checkpoint_path 字段",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行 --checkpoint 优先级高于 YAML 中的配置
    if args.checkpoint is not None:
        cfg["checkpoint_path"] = args.checkpoint

    # ── 构建 DDP 通信环境字典 ─────────────────────────────────────────────────
    env = {
        "master_addr": "localhost",
        "master_port": str(cfg.get("master_port", 29500)),
        "world_size":  cfg["nnodes"] * cfg["nprocs"],
        "group_rank":  int(os.environ.get("SLURM_NODEID", "0")),
    }

    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]
    os.environ["WORLD_SIZE"]  = str(env["world_size"])

    backend = cfg["model"].get("backend", "painn").upper()
    print(f"配置文件：{args.config}")
    print(f"数据路径：{cfg['data']['data_root']}")
    print(f"GPU 数量：{cfg['nprocs']} × 本机 (world_size={env['world_size']})")
    print(f"模型后端：{backend}")

    mp.spawn(
        fn      = train_worker,
        args    = (cfg, env),
        nprocs  = cfg["nprocs"],
        join    = True,
    )

    print("训练完成。")


if __name__ == "__main__":
    main()
