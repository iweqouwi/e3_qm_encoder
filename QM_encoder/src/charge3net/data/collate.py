# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
from typing import List
import torch
import numpy as np
from src.charge3net.data.layer import pad_and_stack


def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    """
    将一个列表（每个元素是一个样本字典）拼合成批次字典。
    字符串/浮点类型的字段（filename, load_time）直接收集为列表，
    其余张量字段通过 pad_and_stack 进行填充和堆叠。
    """
    dict_of_lists = {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}

    pin = (lambda x: x.pin_memory()) if pin_memory else (lambda x: x)

    collated = {}
    for k, v in dict_of_lists.items():
        if k not in ["filename", "load_time"]:
            collated[k] = pin(pad_and_stack(v))
        else:
            collated[k] = v
    return collated
