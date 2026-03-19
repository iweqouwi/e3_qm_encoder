import json
import math
from torch.utils.data import Subset
import numpy as np

def split_data(dataset, val_frac=0.005, split_file=None):
    """
    划分数据集函数。
    
    【形参说明】：
    - dataset (Dataset): PyTorch 的 Dataset 对象，代表完整的原始数据集。
    - val_frac (float): 验证集所占的比例，默认为 0.005（即 0.5%）。只有在不提供 split_file 时才生效。
    - split_file (str): 预先切分好的数据索引文件（通常是 JSON 格式）。如果提供，则忽略 val_frac 随机切分。
    """
    
    # 1. 加载或生成切分索引 (Load or generate splits)
    if split_file is not None:
        # 如果传入了切分文件路径，则打开该 JSON 文件
        with open(split_file, "r") as fp:
            # 读取文件内容。
            # json.load 会返回一个字典，格式通常如：
            # {"train": [0, 2, 3, 5...], "validation": [1, 4, 9...], "test": [8, 11...]}
            splits = json.load(fp)
    else:
        # 如果没有传入切分文件，则根据 val_frac 比例进行随机动态切分
        
        # 获取原数据集的总长度
        datalen = len(dataset)
        
        # 计算验证集应该有多少个样本。
        # 使用 math.ceil (向上取整) 保证即使 val_frac 很小，验证集至少也有 1 个样本。
        num_validation = int(math.ceil(datalen * val_frac))
        
        # np.random.permutation 会生成一个从 0 到 datalen-1 的随机打乱的整数数组
        # 比如 dataset 长度为 10，可能会生成 [3, 1, 8, 4, 0, 9, 2, 7, 5, 6]
        indices = np.random.permutation(len(dataset))
        
        # 利用打乱后的索引数组进行切片，生成一个切分字典
        splits = {
            # 从第 num_validation 个元素一直到最后，全给训练集，并转为 Python 原生 list
            "train": indices[num_validation:].tolist(),
            
            # 取前 num_validation 个元素，给验证集，并转为 Python 原生 list
            "validation": indices[:num_validation].tolist(),
        }

    # 2. 正式切分数据集 (Split the dataset)
    datasplits = {}
    
    # 遍历刚才得到的 splits 字典
    # key 可能是 "train", "validation", "test"
    # indices 是对应集合的索引列表，比如 [0, 2, 3...]
    for key, indices in splits.items():
        # 【核心操作】：利用原数据集 dataset 和 抽取的索引列表 indices，
        # 创建一个 Subset 对象，并存入 datasplits 字典中。
        datasplits[key] = Subset(dataset, indices)
        
    # 返回装有 Subset 对象的字典
    # 使用方可以通过 datasplits["train"] 拿到完整的训练子集
    return datasplits