# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import os
import time
from typing import Union, Optional
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from src.charge3net.data.collate import collate_list_of_dicts
from src.charge3net.data.split import split_data
from src.charge3net.data.graph_construction import GraphConstructor

'''
    数据处理流水线（HDF5版）:
    HDF5 文件 -> DensityDataHDF5 (读取 QM/MM/探针数据) ->
    DensityData (路由到具体读取类) ->
    DensityGraphDataset (子采样探针 + 建图) ->
    Datamodule (分发 DataLoader)

    HDF5 格式:
        /<complex_key>/qm_positions  [N,3] float32
        /<complex_key>/qm_numbers    [N]   int16
        /<complex_key>/mm_positions  [M,3] float32
        /<complex_key>/mm_charges    [M]   float32
        /<complex_key>/probe_xyz     [P,3] float32
        /<complex_key>/probe_target  [P]   float32
'''


class DistributedEvalSampler(torch.utils.data.Sampler):
    """
    不重复、不遗漏地将测试集分发给所有 GPU 的分布式采样器。
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            world = torch.distributed.get_world_size()
        else:
            world = num_replicas
        if rank is None:
            rank = torch.distributed.get_rank()

        self.dataset    = dataset
        self.world      = world
        self.rank       = rank
        self.total_size = len(self.dataset)

        indices          = list(range(self.total_size))
        indices          = indices[self.rank:self.total_size:self.world]
        self.num_samples = len(indices)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.world]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DensityDatamodule:
    """
    QM/MM 电荷密度数据模块。
    负责读取 HDF5 数据、划分 train/val/test 集、构建图并生成 DataLoader。
    """
    def __init__(
        self,
        data_root: Union[str, bytes, os.PathLike],
        graph_constructor,
        num_probes: int = None,
        train_probes: int = 500,
        val_probes: int = 1000,
        test_probes: int = None,
        batch_size: int = 2,
        train_workers: int = 8,
        val_workers: int = 2,
        pin_memory: bool = False,
        val_frac: float = 0.005,
        drop_last: bool = False,
        split_file: Optional[Union[str, bytes, os.PathLike]] = None,
        # 以下参数保留接口兼容性，但对 HDF5 数据集不再使用
        grid_size_file: Optional[Union[str, bytes, os.PathLike]] = None,
        max_grid_construction_size: int = 1000000,
        **kwargs,
    ):
        super().__init__()

        self.data_root   = data_root
        self.batch_size  = batch_size
        self.train_workers = train_workers
        self.val_workers   = val_workers
        self.pin_memory    = pin_memory
        self.val_frac      = val_frac
        self.split_file    = split_file
        self.drop_last     = drop_last

        if num_probes is not None:
            train_probes = val_probes = test_probes = num_probes

        self.train_gc = graph_constructor(num_probes=train_probes)
        self.val_gc   = graph_constructor(num_probes=val_probes)
        self.test_gc  = graph_constructor(num_probes=test_probes)

        dataset = DensityData(self.data_root)
        subsets = split_data(dataset, val_frac=self.val_frac, split_file=self.split_file)

        self.train_set = DensityGraphDataset(subsets["train"],      self.train_gc)
        self.val_set   = DensityGraphDataset(subsets["validation"], self.val_gc)
        # 若 split_file 未提供 test 分区，则用 validation 集代替
        self.test_set  = DensityGraphDataset(
            subsets.get("test", subsets["validation"]), self.test_gc
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batch_size,
            num_workers=self.train_workers,
            sampler=DistributedSampler(self.train_set, drop_last=self.drop_last),
            collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batch_size,
            num_workers=self.val_workers,
            collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.val_workers,
            collate_fn=partial(collate_list_of_dicts, pin_memory=self.pin_memory),
            sampler=DistributedEvalSampler(self.test_set),
        )


class DensityGraphDataset(torch.utils.data.Dataset):
    """
    包装基础数据集，在取数据时调用 graph_constructor 将 QM/MM 原始数据转换为图。
    """
    def __init__(self, dataset, graph_constructor, **kwargs):
        super().__init__(**kwargs)
        self.dataset           = dataset
        self.graph_constructor = graph_constructor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        start_time = time.time()

        data_dict = self.dataset[index]

        # 通过图构建器将原始 QM/MM 数据转为图
        graph_dict = self.graph_constructor(data_dict)

        graph_dict.update(
            filename=data_dict["metadata"]["filename"],
            load_time=time.time() - start_time,
        )
        return graph_dict


class DensityData(torch.utils.data.Dataset):
    """
    数据路由类：根据 datapath 的类型选择合适的底层数据集。
    支持：
      - 单个 HDF5 文件 (.h5 / .hdf5)
      - 文本文件 (.txt)，每行一个 HDF5 文件路径
      - 目录，自动扫描其中的 HDF5 文件
    """
    def __init__(self, datapath, **kwargs):
        super().__init__(**kwargs)
        datapath = str(datapath)
        if os.path.isfile(datapath) and datapath.endswith((".h5", ".hdf5")):
            self.data = DensityDataHDF5(datapath)
        elif os.path.isfile(datapath) and datapath.endswith(".txt"):
            with open(datapath, "r") as f:
                filelist = [
                    os.path.join(os.path.dirname(datapath), line.strip())
                    for line in f if line.strip()
                ]
            self.data = ConcatDataset([DensityData(p) for p in filelist])
        elif os.path.isdir(datapath):
            h5files = sorted([
                os.path.join(datapath, fname)
                for fname in os.listdir(datapath)
                if fname.endswith((".h5", ".hdf5"))
            ])
            if not h5files:
                raise ValueError(f"目录 {datapath} 中未找到任何 .h5/.hdf5 文件")
            self.data = ConcatDataset([DensityDataHDF5(p) for p in h5files])
        else:
            raise ValueError(f"无法识别数据路径 {datapath}，请提供 .h5/.hdf5 文件、.txt 文件列表或含 HDF5 文件的目录")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DensityDataHDF5(torch.utils.data.Dataset):
    """
    从单个 HDF5 文件读取 QM/MM 电荷密度数据。

    HDF5 结构：
        /<complex_key>/
            qm_positions  [N,3] float32
            qm_numbers    [N]   int16
            mm_positions  [M,3] float32
            mm_charges    [M]   float32
            probe_xyz     [P,3] float32
            probe_target  [P]   float32
    """
    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        with h5py.File(filepath, "r") as f: 
            self.member_list = sorted(list(f.keys())) # member_list 是 HDF5 文件中所有复合体的键列表，按字母顺序排序以确保稳定的索引映射
        self.key_to_idx = {k: i for i, k in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.key_to_idx[index]
        key = self.member_list[index]
        with h5py.File(self.filepath, "r") as f:
            grp = f[key]
            qm_positions = grp["qm_positions"][:]
            qm_numbers   = grp["qm_numbers"][:]
            mm_positions = grp["mm_positions"][:]
            mm_charges   = grp["mm_charges"][:]
            probe_xyz    = grp["probe_xyz"][:]
            probe_target = grp["probe_target"][:]
        return {
            "qm_positions": qm_positions,
            "qm_numbers":   qm_numbers,
            "mm_positions": mm_positions,
            "mm_charges":   mm_charges,
            "probe_xyz":    probe_xyz,
            "probe_target": probe_target,
            "metadata":     {"filename": key},
        }
