# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import ase
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from e3nn.util.jit import compile_mode
from torch import nn

import src.charge3net.data.layer as layer


def get_irreps(total_mul, lmax):
    """
    根据给定的总通道数 (total_mul) 和最大角动量 (lmax),均匀分配各个阶数(l)和宇称(p)的通道数。
    Get irreps up to lmax, all with roughly the same multiplicity with a total multiplicity of total_mul
    Example:
        get_irreps(500, lmax=2) = 167x0o + 167x0e + 56x1o + 56x1e + 33x2o + 33x2e
    """
    return [
        (round(total_mul / (lmax + 1) / (l * 2 + 1)), (l, p))
        for l in range(lmax + 1)
        for p in [-1, 1]
    ]


class E3DensityModel(nn.Module):
    """
    顶层模型容器：整合了“原子特征编码器 (E3AtomRepresentationModel)”和“探针消息解码器 (E3ProbeMessageModel)”。
    """
    def __init__(
        self,
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False
    ):
        super().__init__()
        self.spin = spin

        # 1. 实例化原子编码模型：在原子构成的图上进行多次卷积,提取原子的量子化学环境特征
        self.atom_model = E3AtomRepresentationModel(
            num_interactions,
            num_neighbors,
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
        )

        # 2. 实例化探针解码模型：接收原子的特征,通过空间坐标计算任意探针点(Probe)处的标量场(如电子密度)
        self.probe_model = E3ProbeMessageModel(
            num_interactions,
            num_neighbors,
            self.atom_model.atom_irreps_sequence, # 传入原子模型每一层输出的等变特征格式
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
            spin=spin
        )

    def forward(self, input_dict):
        # 步骤 1: 提取多层原子表征
        # atom_representation 是一个列表,包含了经过每一层图卷积后的原子特征张量
        atom_representation = self.atom_model(input_dict)
        
        # 步骤 2: 将原子表征传播到探针点
        # if spin == False, 输出维度: [N_batch, N_probe] (标量电荷密度)
        # if spin == True, 输出维度: [N_batch, N_probe, 2] (预测两个分量)
        probe_result = self.probe_model(input_dict, atom_representation)   
        
        # 如果考虑自旋,将输出解码为自旋向上(spin up)和自旋向下(spin down)的组合
        if self.spin:
            spin_up, spin_down = probe_result[:, :, 0], probe_result[:, :, 1]
            # 重新组合：通道0代表总电荷密度 (up + down),通道1代表自旋密度极化 (up - down)
            probe_result[:, :, 0] = spin_up + spin_down
            probe_result[:, :, 1] = spin_up - spin_down
        return probe_result


class E3AtomRepresentationModel(nn.Module):
    """
    原子表征模型(Encoder)：同质图神经网络,仅在原子节点之间进行消息传递。
    负责编码输入分子的几何结构和原子类别。
    每个隐藏层的不可约表示(及其对应的通道数)是完全一样的(就是这样设计的)
    """
    def __init__(
        self,
        num_interactions, # 消息传递层数(交互层数),每一层都能捕捉更大范围的原子环境
        num_neighbors, # 预期的平均邻居数,用于卷积层中的归一化,通常根据数据集统计得到
        mul=500, # 每个角动量阶数的通道数乘数,控制了模型的整体容量和表达能力
        lmax=4, # 最大角动量阶数,决定了特征表示中包含的标量(l=0)、向量(l=1)、张量(l>=2)的最高阶数
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        
        # 距离的径向基函数扩展网络 (将标量距离 r 映射为高维特征向量)
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # store irreps of each output (mostly so the probe model can use)
        # 记录每一层的特征表示规范(Irreps),供后续的 probe_model 对应使用
        self.atom_irreps_sequence = []

        self.num_species = len(ase.data.atomic_numbers)

        # 初始原子特征：仅仅是 One-hot 编码的原子种类,属于 "0e" (偶宇称标量)
        irreps_node_input = f"{self.num_species}x 0e"  
        # 隐藏层特征维度：由标量(l=0)、向量(l=1)、张量(l>=2)组成的高维等变特征
        # get_irreps决定了每个阶数(l)和宇称(p)的通道数,确保总通道数接近 mul,
        # 分配原则是：每一种阶数 (l) 在底层占据的总浮点数大小是近似相等的(因为每个阶数的特征维度是 2l+1,所以通道数会根据阶数自动调整),
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        # 节点属性：仅用 0e (常数 1) 占位
        irreps_node_attr = "0e"
        # 边的方向特征：空间相对位置向量的球谐函数展开式
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        # 生成距离权重的 MLP 隐藏层配置, number_of_basis在论文中为10
        fc_neurons = [self.number_of_basis, 100]

        # activation to use with even (1) or odd (-1) parities
        # 等变神经网络中,偶宇称特征用普通激活函数,奇宇称特征必须用关于原点奇对称的激活函数(如tanh)
        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input
        '''
            .simplify() 的字面意思是“简化”,它在 e3nn 中的具体作用是: 合并相邻的、完全相同的不可约表示(Irreps)。
            我们得到了 "16x0e + 16x0e + 8x0e"。这三个部分全都是 "0e"(偶宇称标量),可以直接简化成"40e"
            在o3.Irreps() 的括号中, 可以传入列表或者字符串，字符串用加号链接不同的部分"16*0e + 8*1e"，列表则是 [(通道数, (l, p)), ...] 的格式。
            传入元组列表的格式是 [(通道数, (l, p)), ...],其中 l 是阶数,p 是宇称。对于 "0e" 来说,l=0,p=1(偶宇称),所以表示为 (通道数, (0, 1))。
        '''
        # 构建堆叠的等变图卷积层
        for _ in range(num_interactions):
            # 挑选出张量积输出中属于标量(l=0)的部分,用于普通激活
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            
            # 挑选出属于高阶张量(l>0)的部分,这部分不能直接激活,需要使用门控机制(Gate)
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            
            # 为高阶张量生成对应的标量门(Gates),用于乘性调节高阶特征的模长
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            # Gate activation function,用于维持等变性的特殊激活函数模块
            gate = Gate(
                irreps_scalars,  # 标量部分直接激活
                [act[ir.p] for _, ir in irreps_scalars],  # 标量激活函数 返回类似于激活函数的列表：[sigmoid, tanh]
                irreps_gates,    # gates (标量),用于调节高阶张量的模长
                [act_gates[ir.p] for _, ir in irreps_gates],  #  gate 激活函数,必须与 gate 的宇称匹配, [sigmoid, tanh]
                irreps_gated,  # 被门控激活的高阶张量部分,经过 gate 模块后会被乘以对应的 gate 输出,实现非线性调节
            )
            
            # 实例化原子间的等变卷积层
            conv = Convolution(
                irreps_node,          # 输入特征表示
                irreps_node_attr,     # 节点属性表示 ("0e")
                irreps_edge_attr,     # 边方向特征表示 (球谐函数)
                gate.irreps_in,       # 我们期望隐藏层的不可约表示是irreps_node_hidden，但是我们要多给用于后续门控标量,因此直接传入irreps_in，门控层帮我们算好了
                fc_neurons,           # MLP 结构
                num_neighbors,        # 预期的平均邻居数,用于归一化
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

            # 存储每一层的输出 irreps 格式
            self.atom_irreps_sequence.append(irreps_node)  

    def forward(self, input_dict):
        # -- 1. 图结构与边的处理 --
        # 将被 padding 的边索引展平拼接到 Batch 的第 0 维
        '''
            input_dict 中包含了构建图所需的所有信息,其中关键的部分是:
                nodes: [N_batch, N_max_atom] (被 padding 的原子种类索引)
                atom_edges: [N_batch, N_max_edge, 2] (被 padding 的边连接索引,指向原子节点)
                atom_edges_displacement: [N_batch, N_max_edge, 3] (被 padding 的边的位移向量,考虑周期性边界条件)
                num_nodes: [N_batch] (每个图的实际原子数量,用于 unpad_and_cat 展平时正确拼接)
                num_atom_edges: [N_batch] (每个图的实际边数量,用于 unpad_and_cat 展平时正确拼接)
        '''
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )

        # 计算各个子图的边在展平后的全局索引偏移量,以防止不同图的节点混淆
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"]) # 维度: [N_edge, 2]

        edge_src = edges[:, 0] # 源原子索引, 维度: [N_edge]
        edge_dst = edges[:, 1] # 目标原子索引, 维度: [N_edge]

        # -- 2. 节点特征的处理 --
        # 展平原子的空间坐标 XYZ,维度: [N_atom, 3]
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        # 展平原子种类标号,维度: [N_atom]
        nodes = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])

        # 将原子种类转为 one-hot 编码,作为初始特征,维度: [N_atom, num_species]
        nodes = F.one_hot(nodes, num_classes=self.num_species)

        # 创建冗余的节点属性 (全 1),在 e3nn 中常用来承载额外的纯标量信息,维度: [N_atom, 1]
        node_attr = nodes.new_ones(nodes.shape[0], 1)

        # -- 3. 几何特征(边特征)的计算 --
        # 计算相对位移向量 edge_vec,考虑周期性边界条件(PBC)
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        ) # 维度: [N_edge, 3]

        # 对相对位移向量计算球谐函数编码 (编码角度/方向),维度: [N_edge, D_sh]
        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        # 计算边长(标量距离),维度: [N_edge]
        edge_length = edge_vec.norm(dim=1)
        # 通过径向基函数扩展边长特征,维度: [N_edge, D_basis]
        edge_length_embedding = self.basis(edge_length)

        nodes_list = []
        # -- 4. 多层消息传递 (Message Passing) --
        for conv, gate in zip(self.convolutions, self.gates):
            # 执行图卷积更新原子特征
            nodes = conv(
                nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
            )
            # 经过非线性激活层 (门控激活)
            nodes = gate(nodes)
            # 收集每一层的特征,因为后续 Probe 模型需要读取所有层次的局部特征
            nodes_list.append(nodes)

        return nodes_list


class E3ProbeMessageModel(torch.nn.Module):
    """
    探针消息模型(Decoder)：二分图神经网络,仅将消息从【原子】单向传递给【空间探针点】。
    负责根据原子的多层特征场,解码出连续空间内特定坐标下的场强值。
    """
    def __init__(
        self,
        num_interactions,
        num_neighbors,
        atom_irreps_sequence,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=10,
        spin=False
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.basis = RadialBasis(
            start=0.0, 
            end=cutoff,
            number=self.number_of_basis,
            basis=basis,
            cutoff=False,
            normalize=True
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        # Probe 的初始特征也是纯量 "0e",实际输入时全部初始化为0
        irreps_node_input = "0e"
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr = "0e"
        irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons = [self.number_of_basis, 100]

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        irreps_node = irreps_node_input

        for i in range(num_interactions):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )

            # 使用 ConvolutionOneWay,因为这是一个异质图(二分图),源节点(原子)和目标节点(探针)不同
            conv = ConvolutionOneWay(
                irreps_sender_input=atom_irreps_sequence[i], # 发送者：对应第 i 层的原子特征规范
                irreps_sender_attr=irreps_node_attr,
                irreps_receiver_input=irreps_node,           # 接收者：探针点自身的特征规范
                irreps_receiver_attr=irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr,
                irreps_node_output=gate.irreps_in,
                fc_neurons=fc_neurons,
                num_neighbors=num_neighbors,
            )
            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)

        # 最后一层：将高维等变张量特征映射回最终需要预测的物理量(标量场)
        if spin:
            out = "1x0e+1x0o" # 预测两个独立的分量
        else:
            out = "0e"        # 预测单一的总电荷密度
        self.readout = Linear(irreps_node, out)

    def forward(self, input_dict, atom_representation):
        # 展平原子的坐标
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        # 展平探针的空间坐标 XYZ,维度: [N_probe, 3]
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        
        # 计算边的全局索引偏移,逻辑同 Encoder
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        # probe_edges 是从原子指向探针的边索引,维度: [N_p_edge, 2]
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # 计算从原子到探针点的相对位移向量,维度: [N_p_edge, 3]
        probe_edge_vec = calc_edge_vec_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )
        # 计算探针边的球谐特征,维度: [N_p_edge, D_sh]
        probe_edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), probe_edge_vec, True, normalization="component"
        )
        # 探针边的距离径向基编码,维度: [N_p_edge, D_basis]
        probe_edge_length = probe_edge_vec.norm(dim=1)
        probe_edge_length_embedding = self.basis(probe_edge_length)

        probe_edge_src = probe_edges[:, 0] # 发送端：原子索引,维度: [N_p_edge]
        probe_edge_dst = probe_edges[:, 1] # 接收端：探针索引,维度: [N_p_edge]

        # 初始化探针点特征为全零张量,维度: [N_probe, 1]
        probes = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 1),
            device=atom_representation[0].device,
        )

        probe_attr = probes.new_ones(probes.shape[0], 1)
        atom_node_attr = probes.new_ones(atom_xyz.shape[0], 1)

        # Apply interaction layers
        # 这里将探针依次与每一层(多尺度)的原子表征进行一次二分图卷积
        for conv, gate, atom_nodes in zip(
            self.convolutions, self.gates, atom_representation
        ):
            probes = conv(
                atom_nodes,               # Sender特征: 来自 Encoder 该层的原子特征
                atom_node_attr,
                probes,                   # Receiver特征: 当前探针的隐藏层特征
                probe_attr,
                probe_edge_src,
                probe_edge_dst,
                probe_edge_attr,          # 几何方向特征
                probe_edge_length_embedding, # 距离信息
            )
            probes = gate(probes)

        # 最后经过等变线性映射 (Readout),降维为最终需要的标量或自旋密度向量
        probes = self.readout(probes).squeeze() # 维度: [N_probe] 或 [N_probe, 2]

        # 重新将展平的探针结果拆分后按 Batch 形状堆叠回来 (rebatch)
        probes = layer.pad_and_stack(
            torch.split(
                probes,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
        )
        return probes


class RadialBasis(nn.Module):
    r"""
    Wrapper for e3nn.math.soft_one_hot_linspace, with option for normalization
    用于将标量距离(r)扩展为连续的、平滑的类似One-hot或高斯的基函数向量组。
    这使得后续的 MLP 能够更容易地学习距离的非线性映射规律。
    
    Args:
        start (float): 径向基函数的起始值(通常是 0.0)。
        end (float): 径向基函数的结束值(通常等于图的截断半径 cutoff)。
        number (int): 基函数的数量(例如 10,即将距离标量映射为 10 维的向量)。
        basis ({'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}): 
            基函数的类型。默认为 'gaussian'(高斯函数),会在 start 到 end 之间均匀放置 number 个高斯峰。
        cutoff (bool): 是否在边界处强制截断(让区间外的值严格为 0)。
        normalize (bool): 是否对生成的基函数特征进行归一化(使其均值为 0,标准差为 1),这有利于神经网络的梯度稳定。
        samples (int): 如果开启归一化,使用多少个均匀采样点来预估特征的均值和方差。
    """
    def __init__(
        self,
        start,
        end,
        number,
        basis="gaussian",
        cutoff=False,
        normalize=True,
        samples=4000
    ):
        super().__init__()
        # 保存各项超参数
        self.start = start
        self.end = end
        self.number = number
        self.basis = basis
        self.cutoff = cutoff
        self.normalize = normalize

        # 预计算归一化常数(均值和标准差)
        if normalize:
            # 不需要计算梯度
            with torch.no_grad():
                # 在 [start, end] 区间内均匀采样 samples 个点代表可能的距离值 r
                rs = torch.linspace(start, end, samples+1)[1:]
                
                # 调用 e3nn 的底层函数,计算这 samples 个距离点在各个基函数上的激活值
                # bs 的维度将是 [samples, number]
                bs = soft_one_hot_linspace(rs, start, end, number, basis, cutoff)
                assert bs.ndim == 2 and len(bs) == samples
                
                # 沿着样本维度 (dim=0) 计算每一个基函数通道的均值和标准差
                std, mean = torch.std_mean(bs, dim=0)
            
            # 使用 register_buffer 将均值和方差注册为模型的 buffer
            # buffer 不会作为可学习的参数更新,但会随着模型权重 (state_dict) 一起被保存和加载
            self.register_buffer("mean", mean)
            self.register_buffer("inv_std", torch.reciprocal(std)) # 保存标准差的倒数,为了前向传播时用乘法代替除法加速
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入的标量距离张量,通常维度为 [N_edge] 或 [N_edge, 1]
        Returns:
            映射后的径向基特征,维度为 [N_edge, number]
        """
        # 1. 核心映射：将标量距离 x 展开为 number 维的基函数向量
        # 例如 x=2.0 落在第 5 个高斯峰的中心,那么输出向量在第 5 个维度的值接近 1,两边递减
        x = soft_one_hot_linspace(x, self.start, self.end, self.number, self.basis, self.cutoff)
        
        # 2. 归一化：(x - mean) / std,加速模型收敛
        if self.normalize:
            x = (x - self.mean) * self.inv_std
        return x


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    # 辅助函数：根据 Clebsch-Gordan 选择定则,判断特征 in1 和 in2 经过张量积是否能够生成特征 ir_out
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # index 目标原子的索引,即 edge_dst [N_edge]。它记录了每一条边指向哪个原子。
    # src 是边特征张量,维度为 [N_edge, D_mid],包含了每条边的消息信息。
    # 自定义的 scatter_add 函数,功能同 torch_scatter.scatter(dim=0)。
    # 将边特征按目标节点索引累加,实现图神经网络的消息聚合机制(Aggregation)
    out = src.new_zeros(dim_size, src.shape[1]) # 创建一个全零的张量 out,维度是 [N_atom, D_mid], 用于存储每个原子聚合后的特征, 因此N_atom(原子数量) 作为第一维
    index = index.reshape(-1, 1).expand_as(src) # 将边索引展平并扩展到与 src 同维度[N_edge, D_mid],准备进行 scatter_add
    return out.scatter_add_(0, index, src)


def calc_edge_vec(
    positions: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
):
    """
    Calculate vectors of edges
    计算边的三维相对向量,包含对晶体周期性边界条件(PBC)的处理。
    (modified from src.data.layer.calc_distance)

    Args:
        positions: Tensor of shape (num_nodes, 3) with xyz coordinates inside cell
        cells: Tensor of shape (num_splits, 3, 3) with one unit cell for each split
        edges: Tensor of shape (num_edges, 2)
        edges_displacement: Tensor of shape (num_edges, 3) with the offset (in number of cell vectors) of the sending node
        splits: 1-dimensional tensor with the number of edges for each separate graph
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions[edges[:, 1]]  # num_edges, 3
    vec = this_pos - neigh_abs_pos  # num_edges, 3, 从邻居指向自身的向量
    return vec


def calc_edge_vec_to_probe(
    positions: torch.Tensor,
    positions_probe: torch.Tensor,
    cells: torch.Tensor,
    edges: torch.Tensor,
    edges_displacement: torch.Tensor,
    splits: torch.Tensor,
    return_diff=False,
):
    """
    Calculate vectors of edges from atoms to probes
    计算从原子到空间探针点的三维相对向量。
    """
    unitcell_repeat = torch.repeat_interleave(cells, splits, dim=0)  # num_edges, 3, 3
    displacement = torch.matmul(
        torch.unsqueeze(edges_displacement, 1), unitcell_repeat
    )  # num_edges, 1, 3
    displacement = torch.squeeze(displacement, dim=1)
    neigh_pos = positions[edges[:, 0]]  # 发送端原子坐标, num_edges, 3
    neigh_abs_pos = neigh_pos + displacement  # num_edges, 3
    this_pos = positions_probe[edges[:, 1]]  # 接收端探针坐标, num_edges, 3
    vec = this_pos - neigh_abs_pos  # num_edges, 3
    return vec


# Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
# University of California, through Lawrence Berkeley National Laboratory
...
@compile_mode("script")
class Convolution(torch.nn.Module):
    """
    等变卷积层 (同质图版：用于原子之间)
    Equivariant Convolution
    Args:
        irreps_node_input (e3nn.o3.Irreps): representation of the input node features
        irreps_node_attr (e3nn.o3.Irreps): representation of the node attributes
        irreps_edge_attr (e3nn.o3.Irreps): representation of the edge attributes
        irreps_node_output (e3nn.o3.Irreps or None): representation of the output node features
        fc_neurons (list[int]): number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer
        num_neighbors (float): typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        # sc: Self-connection (残差连接网络),负责将自身上一步的信息向下一步传递
        self.sc = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output
        )

        # lin1: 线性变换,用于对要被当成消息发送出去的节点特征做预处理
        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input
        )

        # -- 以下构建核心的张量积网络 (Tensor Product) --
        # 张量积将原子的内部表示(节点特征)和空间方向(边球谐特征)结合起来
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    # 过滤条件：我们只保留需要的输出维度 (self.irreps_node_output 里有的)
                    # 或者保留 "0e" (o3.Irrep(0, 1) 的意思就是 l=0, p=1,即 0e 标量)
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid) # k 是这个新特征在输出列表中的索引位置
                        irreps_mid.append((mul, ir_out)) # 记录输出：继承输入节点的通道数 mul,形状为刚算出来的 ir_out
                        # 添加核心计算指令！
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        # o3.Irreps(irreps_mid).sort()的第一个核心作用就是：把相同物理形状(阶数和宇称相同)的特征挪到一起,并把它们的通道数 (mul) 加起来[也是重载的sort]
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]
        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False, # 权重是由距离 MLP 动态生成的
            shared_weights=False,
        )
        
        # fc: 多层感知机,负责接收径向距离标量,输出用于张量积操作的各个通道耦合权重
        # fc_neurons 的维度为 [number_of_basis, 100]：其中默认的 number_of_basis 是 10,表示将距离扩展为 10 维的基函数特征；100 是 MLP 隐藏层的神经元数量
        # [tp.weight_numel] 是张量积操作中需要的权重数量(计算出来的),MLP 的输出维度必须匹配这个数量,以便为每个张量积通道生成一个权重,这里假设 tp.weight_numel 是 6
        # fc_neurons + [tp.weight_numel] 的拼接结果(两个都是列表,可以这样拼接)是 [10, 100] + [6] = [10, 100, 6]
        # 这告诉后面的网络：我要建一个三层的 MLP,输入 10 维,经过 100 维的隐藏层,最后输出 6 维！

        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        # lin2: 接收邻居聚合来的中间特征,再次做线性映射对齐输出特征维度
        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_node_output
        )
        
        # lin3: 注意力机制/门控的角度生成器,生成一个单通道标量用于控制本层更新比例
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars
    ) -> torch.Tensor:
        # 输入维度:
        # node_input: [N_atom, D_in] 
        # edge_src, edge_dst: [N_edge] 分别存储的是边的源节点和目标节点索引
        # edge_attr: [N_edge, D_sh]
        # edge_scalars: [N_edge, D_basis] D_basis是径向基函数个数,默认是10

        # 1. 根据原子的距离生成动态交互权重 (Radial MLP)
        weight = self.fc(edge_scalars) # 维度: [N_edge, weight_numel]

        # 2. 节点自身的跳跃连接 (Self-Interaction) sc是FullyConnectedTensorProduct,目标不可约表示是irreps_node_output
        node_self_connection = self.sc(node_input, node_attr) # 维度: [N_atom, D_out]
        
        # 3. 对将要发送的消息做初步映射 lin1是FullyConnectedTensorProduct,目标不可约表示是irreps_node_input(保持输入输出维度一致,方便后续张量积计算)
        node_features = self.lin1(node_input, node_attr) # 维度: [N_atom, D_in]

        # 4. 核心张量积交互: 将源节点的化学特征与边的空间方向结合,用距离权重调制,目标不可也表示是irreps_node_output
        # node_features[edge_src] 维度: [N_edge, D_in]
        edge_features = self.tp(node_features[edge_src], edge_attr, weight) 
        

        # 5. 消息聚合 (Aggregation): 将所有指向目标原子的边特征累加
        node_features = scatter(
            edge_features, edge_dst, dim_size=node_input.shape[0]
        ).div(self.num_neighbors**0.5) # 用预估平均邻居数正则化方差。维度: [N_atom, D_mid]

        # 6. 聚合后的信息投影
        node_conv_out = self.lin2(node_features, node_attr) # 维度: [N_atom, D_out]
        
        # 7. 计算一个类似注意力的平滑混合角度,用的是信息投影前的feature进行注意力计算,初始更倾向于保留旧特征,避免训练初期过度更新导致不稳定
        node_angle = 0.1 * self.lin3(node_features, node_attr) # 维度: [N_atom, 1]
        #             ^^^------ start small, favor self-connection (初始更倾向于保留旧特征)

        # 计算正余弦混合权重,平滑合并自连接特征和网络聚合的周边特征
        # 网络在说：“对于标量部分,我有过去的记忆,所以我用正余弦把我的记忆和邻居的新消息平滑融合；
        # 但是对于向量部分,我过去一片空白(全是0),所以我根本不需要插值,我 100% 全盘接收邻居算出来的新向量！
        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        
        # 最终输出维度: [N_atom, D_out]
        return cos * node_self_connection + sin * node_conv_out


@compile_mode("script")
class ConvolutionOneWay(torch.nn.Module):
    """
    单向等变卷积层 (异质/二分图版：用于原子到探针点)
    Equivariant Convolution, but receiving nodes are differently indexed from sending nodes.
    Additionally, sender and receiver nodes can have different irreps.
    """

    def __init__(
        self,
        irreps_sender_input,    # 发送方(原子)特征
        irreps_sender_attr,
        irreps_receiver_input,  # 接收方(探针)特征
        irreps_receiver_attr,
        irreps_edge_attr,
        irreps_node_output,     # 输出特征
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.irreps_sender_input = o3.Irreps(irreps_sender_input)
        self.irreps_sender_attr = o3.Irreps(irreps_sender_attr)
        self.irreps_receiver_input = o3.Irreps(irreps_receiver_input)
        self.irreps_receiver_attr = o3.Irreps(irreps_receiver_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        # 探针自身的连接映射网络
        self.sc = FullyConnectedTensorProduct(
            self.irreps_receiver_input,
            self.irreps_receiver_attr,
            self.irreps_node_output,
        )

        # 原子消息发送的预映射
        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_sender_input, self.irreps_sender_attr, self.irreps_sender_input
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_sender_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        # 张量积：耦合 原子的特征 与 原子到探针的相对方向
        tp = TensorProduct(
            self.irreps_sender_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        # 用距离计算权重的网络
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel], torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, self.irreps_node_output
        )
        self.lin3 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_receiver_attr, "0e"
        )

    def forward(
        self,
        sender_input,   # 维度: [N_atom, D_sender]
        sender_attr,
        receiver_input, # 维度: [N_probe, D_receiver]
        receiver_attr,
        edge_src,       # 维度: [N_p_edge]
        edge_dst,       # 维度: [N_p_edge]
        edge_attr,      # 维度: [N_p_edge, D_sh]
        edge_scalars,   # 维度: [N_p_edge, D_basis]
    ) -> torch.Tensor:
        # 生成距离交互权重
        weight = self.fc(edge_scalars) # 维度: [N_p_edge, weight_numel]

        # 探针原本特征的线性投影映射
        receiver_self_connection = self.sc(receiver_input, receiver_attr) # 维度: [N_probe, D_out]

        # 对原子发送的特征进行线性预处理
        sender_features = self.lin1(sender_input, sender_attr) # 维度: [N_atom, D_sender]

        # 核心步骤：提取目标边源头的原子特征做张量积耦合
        # sender_features[edge_src] 根据边索引获取原子特征, 维度: [N_p_edge, D_sender]
        # 张量积输出维度: [N_p_edge, D_mid]
        edge_features = self.tp(sender_features[edge_src], edge_attr, weight)

        # scatter edge features from sender (atoms) to receiver (probes)
        # 消息聚合：将所有指向当前探针点的信息进行相加,这就好像是空间中的场积分叠加
        # 结果维度: [N_probe, D_mid]
        receiver_features = scatter(
            edge_features, edge_dst, dim_size=receiver_input.shape[0]
        ).div(self.num_neighbors**0.5)

        # 聚合特征向后投影映射
        receiver_conv_out = self.lin2(receiver_features, receiver_attr) # 维度: [N_probe, D_out]
        
        # 计算更新保留门控
        receiver_angle = 0.1 * self.lin3(receiver_features, receiver_attr) # 维度: [N_probe, 1]
        #            ^^^------ start small, favor self-connection

        # 正余弦平滑融合自身信息和接收到的原子扩散场信息
        cos, sin = receiver_angle.cos(), receiver_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        
        # 最终输出维度: [N_probe, D_out]
        return cos * receiver_self_connection + sin * receiver_conv_out


# ============================================================
# ================  量子环境编码器新增组件  ====================
# ============================================================
# 以下所有类构成 Phase-1 量子环境编码器 (QMEnvironmentEncoder) 的完整实现。
# 它们对应《量子环境编码器设计流程.md》中的 4 个主要任务:
#
#   MMPhysicsFeatureComputer   → 任务 1  实时物理场计算
#   QMAtomEncoder              → 任务 1+2 的前置：支持物理注入的等变原子编码器
#   EquivariantFusionBottleneck→ 任务 2  跨层 JK-Attention 融合瓶颈
#   PolynomialEnvelope         → 任务 3 辅助：多项式平滑截断包络函数
#   EnhancedProbeReadoutModel  → 任务 3  单跳增强探针读出网络
#   QMEnvironmentEncoder       → 顶层集成模型
# ============================================================


def filter_irreps_by_l(irreps: o3.Irreps, l: int) -> o3.Irreps:
    """
    辅助函数：从给定的 Irreps 中筛选出角动量阶数恰好为 l 的所有不可约表示。
    例如，对于 "100x0e + 100x0o + 33x1o + 33x1e + 20x2o + 20x2e"，
      filter_irreps_by_l(irreps, l=0) → "100x0e + 100x0o"
      filter_irreps_by_l(irreps, l=1) → "33x1o + 33x1e"
      filter_irreps_by_l(irreps, l=2) → "20x2o + 20x2e"

    Args:
        irreps: 输入的 e3nn Irreps 对象
        l:      目标角动量阶数 (0=标量, 1=向量, 2=张量, ...)

    Returns:
        仅包含阶数为 l 的分量的 Irreps（已 simplify）
    """
    return o3.Irreps(
        [(mul, ir) for mul, ir in irreps if ir.l == l]
    ).simplify()


# ============================================================
# 任务 1 ── 实时物理场特征计算
# ============================================================

class MMPhysicsFeatureComputer(nn.Module):
    """
    实时计算 MM 原子（外部分子力场环境）在 QM 原子位置处产生的静电势和电场。

    物理来源（库仑定律）：
        - 静电势 (l=0 标量): V_i = Σ_{j∈MM}  q_j / ||r_i - r_j||
        - 电场向量 (l=1 矢量): E_i = Σ_{j∈MM}  q_j * (r_i - r_j) / ||r_i - r_j||^3

    注意:
        - 此处省略常数 k_e，由模型后续归一化层隐式学习缩放。
        - 使用 padding mask 而非循环，保证 GPU 上的批量并行性。
        - 加入数值稳定项 eps 防止距离为零时的除零异常。

    Args:
        eps (float): 距离分母的数值稳定小量，默认 1e-8。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pos_QM_padded:     torch.Tensor,   # [B, N_max, 3]  QM 原子坐标（已 padding）
        pos_MM_padded:     torch.Tensor,   # [B, M_max, 3]  MM 原子坐标（已 padding）
        charges_MM_padded: torch.Tensor,   # [B, M_max]     MM 原子偏电荷（已 padding）
        num_mm_atoms:      torch.Tensor,   # [B]            每个样本真实 MM 原子数
    ):
        """
        Returns:
            s_ep : [B, N_max] 每个 QM 原子位置处的静电势标量 (l=0, V_i)
            v_ef : [B, N_max, 3] 每个 QM 原子位置处的电场矢量 (l=1, E_i)
        """
        B, N_max, _ = pos_QM_padded.shape
        M_max = pos_MM_padded.shape[1]

        # ── 步骤 1: 构建 MM 原子 padding mask ──────────────────────────────
        # mm_mask[b, j] = True 表示第 b 个样本的第 j 个 MM 原子是真实原子（非 padding）
        # arange 广播比较，维度: [B, M_max]
        mm_mask = (
            torch.arange(M_max, device=charges_MM_padded.device)[None, :]
            < num_mm_atoms[:, None]
        )

        # 将 padding 位置的电荷置零，确保其不贡献任何静电场量
        # masked_charges: [B, M_max]
        masked_charges = charges_MM_padded * mm_mask.float()

        # ── 步骤 2: 计算 QM 相对于每个 MM 原子的位移向量 ───────────────────
        # r_{ij} = r_QM_i - r_MM_j（即从 MM 原子指向 QM 原子的向量）
        # unsqueeze 广播:
        #   pos_QM_padded.unsqueeze(2)  : [B, N_max, 1, 3]
        #   pos_MM_padded.unsqueeze(1)  : [B, 1, M_max, 3]
        # r_QM_MM: [B, N_max, M_max, 3]
        r_QM_MM = pos_QM_padded.unsqueeze(2) - pos_MM_padded.unsqueeze(1)

        # ── 步骤 3: 计算距离，加 eps 确保数值稳定 ──────────────────────────
        # dist: [B, N_max, M_max]
        dist = torch.norm(r_QM_MM, dim=-1) + self.eps

        # 单位方向向量（从 MM 指向 QM）
        # unit_r: [B, N_max, M_max, 3]
        unit_r = r_QM_MM / dist.unsqueeze(-1)

        # ── 步骤 4: 计算静电势 V_i = Σ_j  q_j / r_ij ───────────────────────
        # masked_charges[:, None, :] : [B, 1, M_max]  →  广播后除以 dist
        # 沿 M_max 维求和: [B, N_max, M_max] → [B, N_max]
        s_ep = (masked_charges[:, None, :] / dist).sum(dim=-1)  # [B, N_max]

        # ── 步骤 5: 计算电场 E_i = Σ_j  q_j * unit_r_ij / r_ij^2 ─────────
        # masked_charges[:, None, :, None] : [B, 1, M_max, 1]
        # dist.unsqueeze(-1)**2            : [B, N_max, M_max, 1]
        # unit_r                           : [B, N_max, M_max, 3]
        # 沿 M_max 维求和: → [B, N_max, 3]
        v_ef = (
            masked_charges[:, None, :, None]
            / dist.unsqueeze(-1) ** 2
            * unit_r
        ).sum(dim=-2)  # [B, N_max, 3]

        # ── 步骤 6: 处理潜在 NaN（QM 和 MM 原子恰好重合的极端情况）─────────
        s_ep = torch.nan_to_num(s_ep, nan=0.0, posinf=0.0, neginf=0.0)
        v_ef = torch.nan_to_num(v_ef, nan=0.0, posinf=0.0, neginf=0.0)

        return s_ep, v_ef


# ============================================================
# 任务 2 前置 ── 支持物理特征注入的等变原子编码器
# ============================================================

class QMAtomEncoder(nn.Module):
    """
    量子原子编码器（Physics-Augmented E3 Atom Encoder）。

    与 E3AtomRepresentationModel 功能相同，核心区别：
        - 初始节点特征从单纯的原子种类 one-hot 编码扩展为：
              one-hot(种类) ⊕ V_i (静电势, 1×0e) ⊕ E_i (电场矢量, 1×1o)
        - 这两个物理偏置特征作为 0e 和 1o 注入等变卷积第一层，使得网络从
          第一层起便能捕捉由 MM 环境决定的长程静电效应。

    与原编码器架构上完全一致，仅 irreps_node_input 和 forward 有差异。
    atom_irreps_sequence 与原架构保持相同接口，供下游融合层使用。
    """

    def __init__(
        self,
        num_interactions: int,    # GNN 层数 K（= Encoder 输出的层数）
        num_neighbors: int,       # 预期平均邻居数，用于归一化
        mul: int = 500,           # 每个 l 的通道数乘数（控制模型容量）
        lmax: int = 4,            # 最大角动量阶数
        cutoff: float = 4.0,      # 截断半径（Å）
        basis: str = "gaussian",  # 径向基函数类型
        num_basis: int = 20,      # 径向基函数数量
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.number_of_basis = num_basis
        self.num_species = len(ase.data.atomic_numbers)  # ASE 中的元素总数 (~119)

        # 径向基函数扩展（将距离标量 r 映射到高维基函数向量）
        self.basis = RadialBasis(
            start=0.0,
            end=cutoff,
            number=num_basis,
            basis=basis,
            cutoff=False,
            normalize=True,
        )

        self.convolutions = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.atom_irreps_sequence = []  # 存储每层输出 irreps，供 EquivariantFusionBottleneck 使用

        # ── 关键改动：初始节点特征包含物理场 ─────────────────────────────────
        # one-hot 原子种类编码     : num_species × 0e（偶宇称标量）
        # 静电势 V_i              : 1 × 0e（标量，库仑势，偶宇称）
        # 电场矢量 E_i            : 1 × 1o（奇宇称矢量，电场方向性）
        #
        # 这样的 irreps 使得第一层卷积能够利用 1o 中的奇宇称信息，
        # 从而在后续层中逐渐生成所有所需的不可约表示（包括 1e, 2o 等）。
        irreps_node_input = o3.Irreps(
            f"{self.num_species}x0e + 1x0e + 1x1o"
        )

        # 隐藏层目标 irreps（每层的输出空间）
        irreps_node_hidden = o3.Irreps(get_irreps(mul, lmax))
        irreps_node_attr   = o3.Irreps("0e")
        irreps_edge_attr   = o3.Irreps.spherical_harmonics(lmax)
        fc_neurons         = [num_basis, 100]

        # 等变激活函数（偶/奇宇称分别使用不同激活函数）
        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        irreps_node = irreps_node_input

        for _ in range(num_interactions):
            # 可以被张量积生成的 l=0 标量分量（用于直接激活）
            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l == 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
            ]).simplify()

            # 可以被生成的 l>0 高阶张量分量（需通过 Gate 门控激活）
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, irreps_edge_attr, ir)
            ])

            # 门控所用的辅助标量（与高阶张量通道数一一对应）
            ir_gate = "0e" if tp_path_exists(irreps_node, irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir_gate) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )

            conv = Convolution(
                irreps_node,
                irreps_node_attr,
                irreps_edge_attr,
                gate.irreps_in,   # 包含额外 gate 标量的目标 irreps
                fc_neurons,
                num_neighbors,
            )

            irreps_node = gate.irreps_out
            self.convolutions.append(conv)
            self.gates.append(gate)
            self.atom_irreps_sequence.append(irreps_node)

    def forward(
        self,
        input_dict: dict,
        s_ep_unpadded: torch.Tensor,   # [N_total, 1]   unpad 后的静电势标量
        v_ef_unpadded: torch.Tensor,   # [N_total, 3]   unpad 后的电场矢量
    ):
        """
        Args:
            input_dict:     原始批次字典（包含 atom_xyz, nodes, cell, 边索引等）
            s_ep_unpadded:  已 unpad 的静电势特征，形状 [N_total, 1]
            v_ef_unpadded:  已 unpad 的电场向量特征，形状 [N_total, 3]

        Returns:
            nodes_list: list of K tensors, nodes_list[k].shape = [N_total, D_hidden_k]
                        每个元素是第 k 层（0-indexed）卷积后的原子特征张量
        """
        # ── 图结构展平 ─────────────────────────────────────────────────────
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat((
                torch.tensor([0], device=input_dict["num_nodes"].device),
                input_dict["num_nodes"][:-1],
            )),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = layer.unpad_and_cat(
            input_dict["atom_edges"] + edge_offset, input_dict["num_atom_edges"]
        )
        edge_src = edges[:, 0]  # [N_edge]
        edge_dst = edges[:, 1]  # [N_edge]

        # ── 节点特征构建 ───────────────────────────────────────────────────
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes    = layer.unpad_and_cat(input_dict["nodes"],    input_dict["num_nodes"])

        # 原子种类 one-hot 编码: [N_total, num_species]
        nodes_onehot = F.one_hot(nodes, num_classes=self.num_species).float()

        # ── 关键：拼接物理场特征，构成增强初始特征 ─────────────────────────
        # one-hot: [N_total, num_species]  (irreps: num_species × 0e)
        # s_ep:    [N_total, 1]            (irreps: 1 × 0e)
        # v_ef:    [N_total, 3]            (irreps: 1 × 1o)
        # 合并后:  [N_total, num_species+1+3] = [N_total, num_species+4]
        nodes = torch.cat([nodes_onehot, s_ep_unpadded, v_ef_unpadded], dim=-1)

        # 节点属性（常数 1，用作 FullyConnectedTensorProduct 的第二输入）
        node_attr = nodes.new_ones(nodes.shape[0], 1)

        # ── 几何特征（边方向与距离）────────────────────────────────────────
        edge_vec = calc_edge_vec(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
        )  # [N_edge, 3]

        # 方向球谐函数特征: [N_edge, D_sh]
        edge_attr = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )
        # 径向距离基函数特征: [N_edge, D_basis]
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.basis(edge_length)

        # ── 多层等变消息传递 ───────────────────────────────────────────────
        nodes_list = []
        for conv, gate in zip(self.convolutions, self.gates):
            nodes = conv(nodes, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
            nodes = gate(nodes)
            nodes_list.append(nodes)   # 收集每层输出供 EquivariantFusionBottleneck 使用

        return nodes_list


# ============================================================
# 任务 2 ── 高级等变特征融合瓶颈层
# ============================================================

class EquivariantFusionBottleneck(nn.Module):
    """
    高级等变特征融合瓶颈（Advanced Equivariant JK-Fusion Bottleneck）。

    替代原来直接取最后一层输出的粗糙做法，通过以下 4 个步骤将 K 层原子
    特征融合为最终的 S_final（标量场）和 V_final（矢量场）：

    Step 4.1 同层解耦与标量化:
        - 提取 l=0 分量 s^(k)，l=1 分量 v^(k)，l=2 分量 T^(k)
        - 对 T^(k) 逐通道计算 Frobenius 范数 n^(k)（旋转不变标量）
        - 拼接 s_hat^(k) = s^(k) ⊕ n^(k)

    Step 4.2 跨层张量积融合:
        - 对相邻层向量场 v^(k) ⊗ v^(k-1) 做 Clebsch-Gordan 张量积
        - l=0 项 p_0^(k) ∝ ⟨v^(k), v^(k-1)⟩ （内积/一致性标量）
        - l=1 项 p_1^(k) ∝ v^(k) × v^(k-1)  （叉积/扭转法向量场）
        - 边界条件 k=0: 无前层，p_0 = p_1 = 0（零张量）

    Step 4.3 等变层级注意力 (JK-Attention):
        - 将各层投影后特征的向量范数与标量特征拼接，输入打分 MLP
        - 在 K 层维度做 Softmax，得到自适应权重 α^(k)

    Step 4.4 统一映射与坍缩:
        - o3.Linear 将各层特征投影到统一隐空间（D_s 标量 + D_v 向量通道）
        - 用注意力权重加权求和，输出 S_final 和 V_final

    Args:
        atom_irreps_sequence: list of K o3.Irreps，每层输出的特征规范
        D_s (int): 输出标量维度（默认 256）
        D_v (int): 输出向量通道数（默认 64，最终有 D_v×1e + D_v×1o 两种宇称）
        score_hidden (int): 注意力打分 MLP 的隐藏层大小（默认 64）
    """

    def __init__(
        self,
        atom_irreps_sequence,
        D_s:          int = 256,
        D_v:          int = 64,
        score_hidden: int = 64,
    ):
        super().__init__()
        self.K  = len(atom_irreps_sequence)
        self.D_s = D_s
        self.D_v = D_v

        # ── 预计算每层的 Irreps 分解 ────────────────────────────────────────
        # 存储每层的 l=0/1/2 子集 Irreps 和对应维度，供 forward 使用
        self._irreps_l0 = []   # l=0 标量 irreps（每层可能不同，如第 0 层缺少 0o）
        self._irreps_l1 = []   # l=1 向量 irreps
        self._irreps_l2 = []   # l=2 张量 irreps
        self._n_ch_l2   = []   # l=2 的总通道数（用于 Frobenius 范数形状推断）
        self._n_ch_v    = []   # l=1 的总向量通道数（用于范数计算 m^(k)）

        for irreps in atom_irreps_sequence:
            irr  = o3.Irreps(irreps)
            il0  = filter_irreps_by_l(irr, 0)
            il1  = filter_irreps_by_l(irr, 1)
            il2  = filter_irreps_by_l(irr, 2)
            # l=2 总通道数 = sum(mul for l=2 components)
            nc2  = sum(mul for mul, _ in il2) if len(il2) > 0 else 0
            # l=1 总通道数 = sum(mul for l=1 components)（每通道 3 个分量）
            ncv  = sum(mul for mul, _ in il1) if len(il1) > 0 else 0
            self._irreps_l0.append(il0)
            self._irreps_l1.append(il1)
            self._irreps_l2.append(il2)
            self._n_ch_l2.append(nc2)
            self._n_ch_v.append(ncv)

        # ── Step 4.1: 各层特征提取投影 ─────────────────────────────────────
        # 使用 o3.Linear 从完整隐藏特征中提取指定 l 的分量
        # 各层独立（因为第 0 层可能缺少 0o/1e/2o 等宇称）
        # 注意：对于标准配置 (lmax≥2)，所有层均有 l=0/1/2 分量。
        # 极端 edge-case（如 lmax=0）下某些投影可能为空，用 nn.Linear(1,1) 占位，
        # 在 forward 中通过 len(irreps) > 0 的 guard 条件跳过实际调用。
        self.proj_l0 = nn.ModuleList()
        self.proj_l1 = nn.ModuleList()
        self.proj_l2 = nn.ModuleList()
        for k in range(self.K):
            irr = o3.Irreps(atom_irreps_sequence[k])
            self.proj_l0.append(
                o3.Linear(irr, self._irreps_l0[k]) if len(self._irreps_l0[k]) > 0
                else nn.Linear(1, 1)   # 占位符，实际上不被调用（forward 中 guard 保护）
            )
            self.proj_l1.append(
                o3.Linear(irr, self._irreps_l1[k]) if len(self._irreps_l1[k]) > 0
                else nn.Linear(1, 1)
            )
            self.proj_l2.append(
                o3.Linear(irr, self._irreps_l2[k]) if len(self._irreps_l2[k]) > 0
                else nn.Linear(1, 1)
            )

        # ── Step 4.2: 跨层张量积 CG 耦合 ───────────────────────────────────
        # 使用 FullyConnectedTensorProduct 计算相邻层向量场的 CG 张量积：
        #   tp_l0: v^(k) ⊗ v^(k-1) → l=0 输出（等价内积，1×0e）
        #   tp_l1: v^(k) ⊗ v^(k-1) → l=1 输出（等价叉积，1×1e）
        # 对 k=0，前一层向量用零代替（无跨层信息），TP 结果自然为零。
        # 所有 k≥1 的层共享相同 irreps_l1，因此可以共用同一个 TP 模块。
        # 取 k=1（第 2 层）的 irreps_l1 作为代表；如果 K=1，则不创建。
        if self.K > 1 and len(self._irreps_l1[1]) > 0:
            il1_ref = self._irreps_l1[1]  # 代表层的 l=1 irreps（所有 k≥1 层相同）
            # 内积 (l=0, 0e)：1o⊗1o→0e, 1e⊗1e→0e 等路径均有效
            self.tp_inner = o3.FullyConnectedTensorProduct(
                il1_ref, il1_ref, o3.Irreps("1x0e")
            )
            # 叉积 (l=1, 1e)：1o⊗1o→1e, 1e⊗1e→1e 等路径均有效
            self.tp_cross = o3.FullyConnectedTensorProduct(
                il1_ref, il1_ref, o3.Irreps("1x1e")
            )
            self._has_cross_layer_tp = True
        else:
            self._has_cross_layer_tp = False

        # ── Step 4.3 + 4.4: 统一投影与注意力机制 ──────────────────────────
        # 输出 irreps：D_v 个 1o 通道 + D_v 个 1e 通道（覆盖两种宇称的向量场）
        # dim(V_final) = D_v*3 + D_v*3 = 6*D_v
        self.irreps_v_out = o3.Irreps(f"{D_v}x1o + {D_v}x1e")
        dim_v_out = self.irreps_v_out.dim   # = 6*D_v

        # 每层的标量投影：(s_hat^(k) ⊕ p_0^(k)) → D_s 维标量
        # 使用普通 nn.Linear（所有输入都是旋转不变标量）
        # p_0 来自 tp_inner（1×0e，dim=1），拼接后输入维度 = dim_l0_k + nc2_k + 1
        # 对 k=0 或无 TP 时，p_0 = 0（dim 依然是 1，用零张量填充）
        self.scalar_proj = nn.ModuleList()
        for k in range(self.K):
            dim_in = self._irreps_l0[k].dim + self._n_ch_l2[k] + 1  # +1 for p_0
            self.scalar_proj.append(nn.Linear(dim_in, D_s))

        # 每层的向量投影：(v^(k) ⊕ p_1^(k)) → irreps_v_out（等变线性映射）
        # p_1 来自 tp_cross（1×1e，dim=3），拼接后输入 irreps = irreps_l1_k + "1x1e"
        # 对 k=0 或无 TP 时，p_1 = 0（"1x1e" 依然存在，用零填充）
        self.vector_proj = nn.ModuleList()
        for k in range(self.K):
            if len(self._irreps_l1[k]) > 0:
                irreps_v_in_k = self._irreps_l1[k] + o3.Irreps("1x1e")
                self.vector_proj.append(o3.Linear(irreps_v_in_k, self.irreps_v_out))
            else:
                # 没有 l=1 分量的特殊情况（lmax=0 时）：占位符，不被调用
                self.vector_proj.append(nn.Linear(1, 1))

        # 注意力打分 MLP
        # 输入：D_s（投影后标量）⊕ 每通道向量范数（2*D_v）
        # 这样注意力在统一的 D_s/D_v 空间中计算，维度一致
        # 两层 MLP → 每层单个标量分数
        dim_score_in = D_s + dim_v_out // 3  # D_s + 2*D_v
        # 【核心修复 4】：为多层注意力机制增加 LayerNorm
        self.score_norm = nn.LayerNorm(dim_score_in)
        # 注意：dim_v_out // 3 = 2*D_v（每个向量通道的范数是标量）
        self.score_mlp = nn.Sequential(
            nn.Linear(dim_score_in, score_hidden),
            nn.SiLU(),
            nn.Linear(score_hidden, 1),
        )

        self.dim_v_out = dim_v_out  # 供外部查询 V_final 的维度

    def forward(self, nodes_list):
        """
        Args:
            nodes_list: list of K tensors，每个形状 [N_total, D_hidden_k]
                        由 QMAtomEncoder.forward() 返回

        Returns:
            S_final : [N_total, D_s]        最终标量原子特征
            V_final : [N_total, 6*D_v]      最终向量原子特征（1o + 1e 混合，连接为平坦张量）
        """
        N = nodes_list[0].shape[0]
        device = nodes_list[0].device

        s_tilde_list = []  # 各层投影后的标量特征: list of [N, D_s]
        v_tilde_list = []  # 各层投影后的向量特征: list of [N, 6*D_v]

        # 保存上一层的 l=1 特征（用于跨层 TP）
        v_prev = None  # 第 0 层没有前层，初始化为 None

        for k in range(self.K):
            feat_k = nodes_list[k]  # [N, D_hidden_k]

            # ── Step 4.1: 解耦与标量化 ─────────────────────────────────────

            # 提取 l=0 标量分量 s^(k): [N, dim_l0_k]
            s_k = self.proj_l0[k](feat_k) if len(self._irreps_l0[k]) > 0 \
                  else torch.zeros(N, 0, device=device)

            # 提取 l=1 向量分量 v^(k): [N, dim_l1_k = n_ch_v_k * 3]
            v_k = self.proj_l1[k](feat_k) if len(self._irreps_l1[k]) > 0 \
                  else torch.zeros(N, 0, device=device)

            # 提取 l=2 张量分量 T^(k): [N, dim_l2_k = n_ch_l2_k * 5]
            T_k = self.proj_l2[k](feat_k) if len(self._irreps_l2[k]) > 0 \
                  else torch.zeros(N, 0, device=device)

            # 对 T^(k) 的每个通道计算 Frobenius 范数 n^(k)
            # l=2 每通道 5 个分量: reshape [N, n_ch_l2_k, 5] → norm → [N, n_ch_l2_k]
            #   n^(k)_c = ||T^(k)_c||_F = sqrt(Σ_{m=-2}^{2} (T^(k)_{c,m})^2)
            if self._n_ch_l2[k] > 0:
                T_k_3d = T_k.reshape(N, self._n_ch_l2[k], 5)  # [N, C_2, 5]
                n_k = T_k_3d.norm(dim=-1)                       # [N, C_2]
            else:
                n_k = torch.zeros(N, 0, device=device)

            # s_hat^(k) = s^(k) ⊕ n^(k)（增强标量：包含了高阶张量的旋转不变信息）
            # 形状: [N, dim_l0_k + n_ch_l2_k]
            s_hat_k = torch.cat([s_k, n_k], dim=-1)

            # ── Step 4.2: 跨层张量积 ──────────────────────────────────────
            # p_0^(k): 相邻层向量场的 CG 内积（l=0 标量，旋转不变）
            # p_1^(k): 相邻层向量场的 CG 叉积（l=1 矢量，扭转方向）
            # 边界条件 k=0：无前层，p_0 = 0，p_1 = 零矢量

            # p_0^(k) 的维度: [N, 1]（1×0e）
            # p_1^(k) 的维度: [N, 3]（1×1e）
            if (
                k > 0
                and self._has_cross_layer_tp
                and v_prev is not None
                and v_k.shape[-1] > 0
            ):
                # v_k 和 v_prev 共享相同的 irreps（从 k=1 开始所有层相同）
                # FullyConnectedTensorProduct 要求输入维度匹配已注册的 irreps
                p_0_k = self.tp_inner(v_k, v_prev)   # [N, 1]  (1×0e)
                p_1_k = self.tp_cross(v_k, v_prev)   # [N, 3]  (1×1e)
            else:
                # k=0 或无 TP：用零填充，使网络从第一层无跨层耦合开始训练
                p_0_k = torch.zeros(N, 1, device=device)
                p_1_k = torch.zeros(N, 3, device=device)

            # 更新 v_prev（仅从 k=1 开始有意义，且需要形状与 tp_inner 匹配）
            if self._has_cross_layer_tp and v_k.shape[-1] > 0:
                if k == 0:
                    # 第 0 层向量可能少于后续层（缺少某些宇称），
                    # 但不用于 TP（边界条件已处理），仅保存供 k=1 使用。
                    # 注意：若第 0 层 irreps_l1 与代表层不同，TP 不能直接调用，
                    # 因此只在 k≥1 时启用 TP（上面的 k>0 判断已保证）。
                    pass
                v_prev = v_k   # 保存当前层向量，供下一层 TP 使用

            # ── Step 4.4: 统一投影 ──────────────────────────────────────
            # 标量投影: (s_hat^(k) ⊕ p_0^(k)) → [N, D_s]
            scalar_in_k = torch.cat([s_hat_k, p_0_k], dim=-1)   # [N, dim_l0_k + n_ch_l2_k + 1]
            s_tilde_k = self.scalar_proj[k](scalar_in_k)         # [N, D_s]

            # 向量投影: (v^(k) ⊕ p_1^(k)) → [N, 6*D_v]（等变 o3.Linear）
            if len(self._irreps_l1[k]) > 0 and v_k.shape[-1] > 0:
                vector_in_k = torch.cat([v_k, p_1_k], dim=-1)    # [N, dim_l1_k + 3]
                v_tilde_k   = self.vector_proj[k](vector_in_k)    # [N, 6*D_v]
            else:
                v_tilde_k = torch.zeros(N, self.dim_v_out, device=device)

            s_tilde_list.append(s_tilde_k)
            v_tilde_list.append(v_tilde_k)

        # ── Step 4.3: JK-Attention 打分（在投影后的统一空间中计算）─────────
        # 对每层投影后的向量特征计算每个通道的 L2 范数（旋转不变标量）
        # v_tilde_k: [N, 6*D_v]，reshape 到 [N, 2*D_v, 3]，再求 norm → [N, 2*D_v]
        n_v_channels = self.dim_v_out // 3    # = 2*D_v（1o 通道 + 1e 通道）
        scores = []
        for k in range(self.K):
            v_tilde_k   = v_tilde_list[k]                                    # [N, 6*D_v]
            m_k         = v_tilde_k.reshape(N, n_v_channels, 3).norm(dim=-1) # [N, 2*D_v]
            score_in_k  = torch.cat([s_tilde_list[k], m_k], dim=-1)         # [N, D_s+2*D_v]
            # 【核心】：执行注意力前归一化，防止 Softmax 梯度饱和
            score_in_k  = self.score_norm(score_in_k)
            # e^(k) = MLP(s_tilde^(k) ⊕ m^(k)): [N, 1]
            e_k         = self.score_mlp(score_in_k)                         # [N, 1]
            scores.append(e_k)

        # Stack 并 Softmax: [N, K, 1] → alpha: [N, K, 1]
        E = torch.stack(scores, dim=1)           # [N, K, 1]
        alpha = torch.softmax(E, dim=1)          # [N, K, 1]，沿层维度归一化

        # ── 加权求和，得到最终特征 ─────────────────────────────────────────
        # S_final = Σ_k α^(k) * s_tilde^(k)
        S_stack = torch.stack(s_tilde_list, dim=1)  # [N, K, D_s]
        S_final  = (alpha * S_stack).sum(dim=1)      # [N, D_s]

        # V_final = Σ_k α^(k) * v_tilde^(k)
        V_stack = torch.stack(v_tilde_list, dim=1)  # [N, K, 6*D_v]
        V_final  = (alpha * V_stack).sum(dim=1)      # [N, 6*D_v]

        return S_final, V_final


# ============================================================
# 任务 3 辅助 ── 多项式平滑截断包络函数
# ============================================================

class PolynomialEnvelope(nn.Module):
    """
    DimeNet 风格的多项式平滑截断包络函数 C(d)。

    保证在截断半径 r_cut 处能量曲线连续（值和导数均为 0），避免在截断边界
    引入不连续的力。

    公式 (p=5):
        x = d / r_cut
        C(d) = 1 - (p+1)(p+2)/2 * x^p + p(p+2) * x^{p+1} - p(p+1)/2 * x^{p+2}
        对 d >= r_cut：C(d) = 0

    等价于: C(d) = 1 - 21*x^5 + 35*x^6 - 15*x^7  (p=5)

    Args:
        cutoff (float): 截断半径 r_cut（与图构建截断一致）
        p      (int):   多项式阶数（默认 5，DimeNet 建议值）
    """

    def __init__(self, cutoff: float, p: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
        # 预计算多项式系数
        self.c0 = 1.0
        self.c1 = -((p + 1) * (p + 2)) / 2.0   # = -21  (p=5)
        self.c2 = p * (p + 2)                   # = +35  (p=5)
        self.c3 = -(p * (p + 1)) / 2.0          # = -15  (p=5)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d: 距离张量，任意形状

        Returns:
            envelope 值，形状与 d 相同，d >= cutoff 处为 0
        """
        x = d / self.cutoff                                             # 归一化距离
        x_p   = x.pow(self.p)
        x_p1  = x.pow(self.p + 1)
        x_p2  = x.pow(self.p + 2)
        env = (
            self.c0
            + self.c1 * x_p
            + self.c2 * x_p1
            + self.c3 * x_p2
        )
        # 在截断半径外严格置零（考虑浮点误差使用 < 而非 <=）
        env = env * (d < self.cutoff).float()
        return env


# ============================================================
# 任务 3 ── 原子-探针单向增强读出网络
# ============================================================

class EnhancedProbeReadoutModel(nn.Module):
    """
    增强型单向原子-探针读出网络（Enhanced One-way Probe Readout）。

    这是"信息瓶颈"的核心模块：通过严格的单跳（single-hop）消息传递迫使上游
    编码器必须将完整的量子化学信息压缩至 S_final 和 V_final 中。

    Step 5.1 空间几何不变量:
        - RBF 距离编码 e_{pi} = RBF(d_{pi})
        - 方向性内积（核心）: q_{pi} = ⟨r̂_{pi}, V_{final,i}⟩  [每向量通道一个标量]
          其中 r̂_{pi} = (r_p - r_i) / d_{pi}，为从原子指向探针的单位向量
          这是一个旋转不变量：同时旋转 r̂_{pi} 和 V_{final,i} 结果不变
        - 向量场无方向强度 n_i = ||V_{final,i}||_2  [每通道]

    Step 5.2 宽幅消息生成:
        m_{pi} = MLP_msg(S_{final,i} ⊕ n_i ⊕ e_{pi} ⊕ q_{pi})

    Step 5.3 包络加权聚合:
        m_p = Σ_{i∈N(p)} m_{pi} · C(d_{pi})

    Step 5.4 密度预测:
        ρ_p = MLP_out(m_p)

    Args:
        D_s         (int): S_final 的标量维度（来自 EquivariantFusionBottleneck）
        D_v_channels(int): V_final 的向量通道数（2*D_v，= D_v_1o + D_v_1e）
        D_rbf       (int): 距离 RBF 特征维度
        cutoff      (float): 图截断半径（Å），与图构建一致
        msg_hidden  (int): MLP_msg 隐藏层维度
        out_hidden  (int): MLP_out 隐藏层维度
        spin        (bool): 是否预测自旋密度（双通道输出）
    """

    def __init__(
        self,
        D_s:          int   = 256,   # S_final 的标量维度
        D_v_channels: int   = 128,   # V_final 的向量通道数（= 2*D_v）
        D_rbf:        int   = 20,    # 探针-原子距离的 RBF 维度
        cutoff:       float = 4.0,   # 截断半径（Å）
        msg_hidden:   int   = 256,   # MLP_msg 隐藏层大小
        out_hidden:   int   = 128,   # MLP_out 隐藏层大小
        spin:         bool  = False, # 是否预测自旋密度
    ):
        super().__init__()
        self.cutoff = cutoff
        self.spin   = spin
        self.D_v_channels = D_v_channels

        # ── RBF 距离编码层（径向基函数）──────────────────────────────────
        # 将探针-原子距离 d_{pi} 映射为 D_rbf 维特征向量
        self.rbf = RadialBasis(
            start=0.0,
            end=cutoff,
            number=D_rbf,
            basis="gaussian",
            cutoff=False,
            normalize=True,
        )

        # ── 多项式截断包络函数 ────────────────────────────────────────────
        self.envelope = PolynomialEnvelope(cutoff=cutoff, p=5)

        # ── MLP_msg: 消息生成网络 ─────────────────────────────────────────
        # 输入特征拼接：
        #   S_{final,i}   : [D_s]       原子标量特征
        #   n_i           : [D_v_channels]  向量场强度（旋转不变）
        #   e_{pi}        : [D_rbf]     距离 RBF 编码
        #   q_{pi}        : [D_v_channels]  方向内积（旋转不变）
        dim_msg_in = D_s + D_v_channels + D_rbf + D_v_channels
        # 增加 LayerNorm 抹平物理不变量的均值漂移
        self.msg_norm = nn.LayerNorm(dim_msg_in)
        self.mlp_msg = nn.Sequential(
            nn.Linear(dim_msg_in, msg_hidden),
            nn.SiLU(),
            nn.Linear(msg_hidden, msg_hidden),
            nn.SiLU(),
        )

        # ── MLP_out: 密度预测网络 ─────────────────────────────────────────
        # 输入：聚合后的消息 [msg_hidden]
        # 输出：标量密度 [1] 或自旋密度 [2]
        n_out = 2 if spin else 1
        self.mlp_out = nn.Sequential(
            nn.Linear(msg_hidden, out_hidden),
            nn.SiLU(),
            nn.Linear(out_hidden, n_out),
        )
        # 强制将最后一步线性输出层的偏置初始化为 -10.0，使其默认输出极小的对数密度
        nn.init.constant_(self.mlp_out[-1].bias, -10.0)
        # 强制清零权重，彻底消除初始阶段的方差噪音！
        nn.init.zeros_(self.mlp_out[-1].weight)
    def forward(
        self,
        input_dict:  dict,
        S_final:     torch.Tensor,   # [N_total, D_s]
        V_final:     torch.Tensor,   # [N_total, D_v_channels*3]
    ):
        """
        Args:
            input_dict: 批次字典（包含 probe_xyz, atom_xyz, probe_edges 等）
            S_final:    来自 EquivariantFusionBottleneck 的原子标量特征
            V_final:    来自 EquivariantFusionBottleneck 的原子向量特征（平坦化）

        Returns:
            probes: [B, P] 或 [B, P, 2]（spin）的预测密度值
        """
        # ── 展平坐标与边索引 ──────────────────────────────────────────────
        atom_xyz  = layer.unpad_and_cat(input_dict["atom_xyz"],  input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(input_dict["probe_xyz"], input_dict["num_probes"])

        # 边索引偏移（多图展平时防止不同图节点混淆）
        edge_offset = torch.cumsum(
            torch.cat((
                torch.tensor([0], device=input_dict["num_nodes"].device),
                input_dict["num_nodes"][:-1],
            )),
            dim=0,
        )
        probe_offset = torch.cumsum(
            torch.cat((
                torch.tensor([0], device=input_dict["num_probes"].device),
                input_dict["num_probes"][:-1],
            )),
            dim=0,
        )
        edge_probe_offset = torch.stack(
            [edge_offset, probe_offset], dim=-1
        )[:, None, :]   # [B, 1, 2]

        probe_edges = layer.unpad_and_cat(
            input_dict["probe_edges"] + edge_probe_offset,
            input_dict["num_probe_edges"],
        )  # [N_pe, 2]：probe_edges[:,0]=原子索引, [:,1]=探针索引

        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )

        edge_src = probe_edges[:, 0]   # 原子端（sender）索引 [N_pe]
        edge_dst = probe_edges[:, 1]   # 探针端（receiver）索引 [N_pe]

        # ── Step 5.1: 空间几何不变量计算 ─────────────────────────────────

        # 从原子到探针的相对向量（沿 probe-atom 方向）
        probe_edge_vec = calc_edge_vec_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
        )  # [N_pe, 3]，从原子指向探针

        # 探针-原子距离
        d_pi = probe_edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N_pe, 1]
        d_pi_scalar = d_pi.squeeze(-1)                                     # [N_pe]

        # 从原子指向探针的单位方向向量 r̂_{pi}
        r_hat_pi = probe_edge_vec / d_pi    # [N_pe, 3]

        # RBF 距离编码 e_{pi}
        e_pi = self.rbf(d_pi_scalar)        # [N_pe, D_rbf]

        # 包络权重 C(d_{pi})（用于后续聚合加权）
        C_pi = self.envelope(d_pi_scalar)   # [N_pe]

        # 根据边索引从原子特征中取出发送端特征
        S_i = S_final[edge_src]             # [N_pe, D_s]
        V_i = V_final[edge_src]             # [N_pe, D_v_channels * 3]

        # 将向量特征 reshape 为 [N_pe, D_v_channels, 3]（每通道一个 3D 向量）
        V_i_3d = V_i.reshape(V_i.shape[0], self.D_v_channels, 3)

        # 向量场无方向强度 n_i = ||V_{final,i}||_2（逐通道范数，旋转不变）
        # n_i: [N_pe, D_v_channels]
        n_i = torch.sqrt((V_i_3d ** 2).sum(dim=-1) + 1e-8)

        # 方向性内积 q_{pi} = ⟨r̂_{pi}, V_{final,i}⟩（逐通道点积，旋转不变）
        # r_hat_pi.unsqueeze(1): [N_pe, 1, 3]
        # * V_i_3d:              [N_pe, D_v_channels, 3]
        # .sum(-1):              [N_pe, D_v_channels]
        q_pi = (r_hat_pi.unsqueeze(1) * V_i_3d).sum(dim=-1)   # [N_pe, D_v_channels]

        # ── Step 5.2: 宽幅消息生成 ────────────────────────────────────────
        # m_{pi} = MLP_msg(S_{final,i} ⊕ n_i ⊕ e_{pi} ⊕ q_{pi})
        # 输入维度: D_s + D_v_channels + D_rbf + D_v_channels
        msg_input = torch.cat([S_i, n_i, e_pi, q_pi], dim=-1)   # [N_pe, dim_msg_in]
        # 【核心应用 1】：对拼接后的特征执行层归一化，救活 SiLU 激活函数，稳定训练初期的数值范围，避免 NaN 爆炸！
        msg_input = self.msg_norm(msg_input)
        m_pi = self.mlp_msg(msg_input)                            # [N_pe, msg_hidden]

        # ── Step 5.3: 包络加权聚合 ────────────────────────────────────────
        # m_p = Σ_{i∈N(p)} m_{pi} · C(d_{pi})
        # C_pi.unsqueeze(-1): [N_pe, 1]，广播乘到消息每个维度
        m_pi_weighted = m_pi * C_pi.unsqueeze(-1)                 # [N_pe, msg_hidden]

        num_probes = int(probe_xyz.shape[0])
        # scatter_add 将所有来自同一探针的消息累加, 增加 / (20.0 ** 0.5) 缩放因子，控制聚合后的方差
        m_p = scatter(m_pi_weighted, edge_dst, dim_size=num_probes) / (20.0 ** 0.5) # [N_probe, msg_hidden]

        # ── Step 5.4: 密度预测 ────────────────────────────────────────────
        rho_p = self.mlp_out(m_p)      # [N_probe, 1] 或 [N_probe, 2]
        rho_p = rho_p.squeeze(-1)      # [N_probe] 或 [N_probe, 2]（spin）

        # 将展平的探针结果按 batch 大小重新堆叠为 [B, P] 或 [B, P, 2]
        probes = layer.pad_and_stack(
            torch.split(
                rho_p,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
        )
        return probes


# ============================================================
# 顶层集成 ── 量子环境编码器
# ============================================================

class QMEnvironmentEncoder(nn.Module):
    """
    量子环境编码器（Quantum Environment Encoder, QME-Encoder）。

    Phase-1 预训练模型的顶层容器，整合四个子模块：

    1. MMPhysicsFeatureComputer
       → 实时计算 MM 环境的静电势和电场，作为物理偏置注入原子初始特征。

    2. QMAtomEncoder
       → 支持物理特征注入的等变图卷积编码器，输出 K 层多尺度原子特征。

    3. EquivariantFusionBottleneck
       → 跨层 CG 张量积 + JK-Attention，将 K 层特征融合为 S_final 和 V_final。
       目的：建立"信息瓶颈"，迫使 S_final/V_final 压缩完整的量子化学信息。

    4. EnhancedProbeReadoutModel
       → 单跳方向性消息传递，利用 S_final/V_final 预测探针位置处的电子密度 ρ。

    Args:
        num_interactions (int): 等变卷积层数 K（默认 6）
        num_neighbors    (int): 预期平均邻居数，用于卷积归一化
        mul              (int): 等变特征通道数乘数（控制模型容量，默认 500）
        lmax             (int): 球谐函数最大角动量阶数（默认 4）
        cutoff           (float): 图截断半径（Å，默认 4.0）
        basis            (str): 径向基函数类型（默认 "gaussian"）
        num_basis        (int): 径向基函数数量（默认 10）
        D_s              (int): 融合瓶颈输出标量维度（默认 256）
        D_v              (int): 融合瓶颈输出向量通道数（默认 64，
                                最终 V_final 有 2*D_v 个通道，覆盖两种宇称）
        D_rbf            (int): 探针读出层 RBF 维度（默认 20）
        probe_msg_hidden (int): 探针消息 MLP 隐藏维度（默认 256）
        probe_out_hidden (int): 探针输出 MLP 隐藏维度（默认 128）
        score_hidden     (int): 融合瓶颈注意力 MLP 隐藏维度（默认 64）
        spin             (bool): 是否预测自旋密度（默认 False）
        physics_eps      (float): 静电计算数值稳定小量（默认 1e-8）
    """

    def __init__(
        self,
        num_interactions: int   = 6,
        num_neighbors:    int   = 20,
        mul:              int   = 500,
        lmax:             int   = 4,
        cutoff:           float = 4.0,
        basis:            str   = "gaussian",
        num_basis:        int   = 10,
        D_s:              int   = 256,
        D_v:              int   = 64,
        D_rbf:            int   = 20,
        probe_msg_hidden: int   = 256,
        probe_out_hidden: int   = 128,
        score_hidden:     int   = 64,
        spin:             bool  = False,
        physics_eps:      float = 1e-8,
    ):
        super().__init__()
        self.spin = spin

        # ── 子模块 1: 实时物理场计算器 ──────────────────────────────────
        self.physics_computer = MMPhysicsFeatureComputer(eps=physics_eps)

        # ── 子模块 2: 物理注入式原子编码器 ──────────────────────────────
        self.atom_model = QMAtomEncoder(
            num_interactions=num_interactions,
            num_neighbors=num_neighbors,
            mul=mul,
            lmax=lmax,
            cutoff=cutoff,
            basis=basis,
            num_basis=num_basis,
        )

        # ── 子模块 3: 等变特征融合瓶颈 ──────────────────────────────────
        self.fusion_bottleneck = EquivariantFusionBottleneck(
            atom_irreps_sequence=self.atom_model.atom_irreps_sequence,
            D_s=D_s,
            D_v=D_v,
            score_hidden=score_hidden,
        )

        # V_final 的向量通道总数（= 2*D_v，两种宇称各 D_v 个通道）
        D_v_total_channels = D_v * 2  # 1o 通道 + 1e 通道

        # ── 子模块 4: 单跳增强探针读出网络 ──────────────────────────────
        self.probe_model = EnhancedProbeReadoutModel(
            D_s=D_s,
            D_v_channels=D_v_total_channels,
            D_rbf=D_rbf,
            cutoff=cutoff,
            msg_hidden=probe_msg_hidden,
            out_hidden=probe_out_hidden,
            spin=spin,
        )

    def forward(self, input_dict: dict):
        """
        整合前向传播：

        1. 从 MM 环境计算静电势（V_i）和电场（E_i），作为物理先验注入 QM 节点初始特征。
        2. 多层等变消息传递（K 层 QMAtomEncoder），输出多尺度原子表征列表。
        3. 跨层 JK-Attention 融合，输出 S_final [N_total, D_s] 和 V_final [N_total, 6*D_v]。
        4. 单跳探针读出，输出预测密度 ρ_p。

        Args:
            input_dict: 批次字典，至少包含：
                'atom_xyz'            [B, N_max, 3]
                'nodes'               [B, N_max]
                'num_nodes'           [B]
                'atom_edges'          [B, E_max, 2]
                'atom_edges_displacement' [B, E_max, 3]
                'num_atom_edges'      [B]
                'mm_positions'        [B, M_max, 3]
                'mm_charges'          [B, M_max]
                'num_mm_atoms'        [B]
                'probe_xyz'           [B, P_max, 3]
                'probe_edges'         [B, PE_max, 2]
                'probe_edges_displacement' [B, PE_max, 3]
                'num_probes'          [B]
                'num_probe_edges'     [B]
                'cell'                [B, 3, 3]

        Returns:
            probe_result: [B, P] 或 [B, P, 2]（spin=True 时）
        """
        # ── Step 1: 实时物理场特征计算 ──────────────────────────────────
        # 输入为 padded 批次张量，输出为 padded 的 (s_ep, v_ef)
        # s_ep: [B, N_max]    静电势标量
        # v_ef: [B, N_max, 3] 电场矢量
        s_ep_padded, v_ef_padded = self.physics_computer(
            pos_QM_padded     = input_dict["atom_xyz"],
            pos_MM_padded     = input_dict["mm_positions"],
            charges_MM_padded = input_dict["mm_charges"],
            num_mm_atoms      = input_dict["num_mm_atoms"],
        )

        # 展平 (unpad) 为 [N_total, 1] 和 [N_total, 3]，供 QMAtomEncoder 使用
        # s_ep_padded.unsqueeze(-1): [B, N_max, 1]，满足 unpad_and_cat 的输入格式
        s_ep_unpadded = layer.unpad_and_cat(
            s_ep_padded.unsqueeze(-1), input_dict["num_nodes"]
        )  # [N_total, 1]

        v_ef_unpadded = layer.unpad_and_cat(
            v_ef_padded, input_dict["num_nodes"]
        )  # [N_total, 3]

        # ── Step 2: 多层等变原子编码 ────────────────────────────────────
        # nodes_list[k]: [N_total, D_hidden_k]，k = 0..K-1
        nodes_list = self.atom_model(
            input_dict    = input_dict,
            s_ep_unpadded = s_ep_unpadded,
            v_ef_unpadded = v_ef_unpadded,
        )

        # ── Step 3: 等变特征融合瓶颈 ────────────────────────────────────
        # S_final: [N_total, D_s]
        # V_final: [N_total, 6*D_v]  (D_v 个 1o 通道 + D_v 个 1e 通道，连接平坦)
        S_final, V_final = self.fusion_bottleneck(nodes_list)

        # ── Step 4: 增强型单跳探针读出 ──────────────────────────────────
        # 输出: [B, P] 或 [B, P, 2]（spin）
        probe_result = self.probe_model(
            input_dict = input_dict,
            S_final    = S_final,
            V_final    = V_final,
        )

        # 自旋密度分解（与原 E3DensityModel 保持一致接口）
        if self.spin:
            spin_up   = probe_result[:, :, 0]
            spin_down = probe_result[:, :, 1]
            probe_result[:, :, 0] = spin_up + spin_down   # 总密度
            probe_result[:, :, 1] = spin_up - spin_down   # 极化密度

        return probe_result