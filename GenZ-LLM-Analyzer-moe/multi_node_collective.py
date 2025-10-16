"""
多节点通信算子模块
区分节点内和节点间的通信开销
"""


def get_intra_node_AR_time(data, num_gpus, system):
    """
    节点内All-Reduce时间（张量并行）
    使用节点内的PCIe连接
    
    Args:
        data: 每个GPU的消息大小(Bytes)
        num_gpus: 节点内GPU数量
        system: 节点内System对象（配置了PCIe带宽和延迟）
        
    Returns:
        time: All-Reduce完成时间(ms)
    """
    if data == 0 or num_gpus == 1:
        return 0
    
    # Ring All-Reduce: 启动延迟 + 链路延迟 + 数据传输时间
    # 公式: Start Latency + N*Tlink + 2M*(N-1)/(N*BW)
    allReduceTime = (
        5e-6 + 
        2 * (num_gpus - 1) * system.interchip_link_latency + 
        2 * (num_gpus - 1) * (data / num_gpus) / system.interchip_link_bw
    ) * 1000
    
    return allReduceTime


def get_inter_node_P2P_time(data, system):
    """
    节点间点对点通信时间（流水线并行）
    使用节点间的InfiniBand连接
    
    Args:
        data: 消息大小(Bytes)
        system: 节点间System对象（配置了InfiniBand带宽和延迟）
        
    Returns:
        time: 点对点传输完成时间(ms)
    """
    if data == 0:
        return 0
    
    # P2P传输: 链路延迟 + 数据传输时间
    p2p_time = (
        system.interchip_link_latency + 
        data / system.interchip_link_bw
    ) * 1000
    
    return p2p_time


def get_intra_node_AG_time(data, num_gpus, system):
    """
    节点内All-Gather时间
    
    Args:
        data: 每个GPU的消息大小(Bytes)
        num_gpus: 节点内GPU数量
        system: 节点内System对象
        
    Returns:
        time: All-Gather完成时间(ms)
    """
    if data == 0 or num_gpus == 1:
        return 0
    
    # Ring All-Gather
    allGatherTime = (
        5e-6 + 
        (num_gpus - 1) * system.interchip_link_latency + 
        (num_gpus - 1) * (data / num_gpus) / system.interchip_link_bw
    ) * 1000
    
    return allGatherTime


def get_communication_time(
    comm_type,
    data_size,
    is_intra_node,
    intra_node_system,
    inter_node_system,
    num_gpus=2
):
    """
    统一的通信时间计算接口
    
    Args:
        comm_type: 通信类型 ('allreduce', 'p2p', 'allgather')
        data_size: 数据大小(Bytes)
        is_intra_node: 是否为节点内通信
        intra_node_system: 节点内System配置
        inter_node_system: 节点间System配置
        num_gpus: GPU数量（用于集合通信）
        
    Returns:
        time: 通信时间(ms)
    """
    if is_intra_node:
        # 节点内通信（PCIe）
        if comm_type == 'allreduce':
            return get_intra_node_AR_time(data_size, num_gpus, intra_node_system)
        elif comm_type == 'allgather':
            return get_intra_node_AG_time(data_size, num_gpus, intra_node_system)
        else:
            raise ValueError(f"不支持的节点内通信类型: {comm_type}")
    else:
        # 节点间通信（InfiniBand）
        if comm_type == 'p2p':
            return get_inter_node_P2P_time(data_size, inter_node_system)
        else:
            raise ValueError(f"不支持的节点间通信类型: {comm_type}")
