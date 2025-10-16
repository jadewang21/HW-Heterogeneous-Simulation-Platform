"""
多节点系统配置模块
支持节点内（intra-node）和节点间（inter-node）通信的区分
"""

import numpy as np
from GenZ.system import System
from GenZ.unit import Unit


class MultiNodeSystem:
    """
    多节点系统配置类
    
    用于配置跨节点的分布式系统，区分：
    - 节点内通信：通过PCIe Gen4连接的GPU之间的通信
    - 节点间通信：通过InfiniBand等高速网络连接的节点间通信
    """
    
    def __init__(
        self,
        num_nodes=2,
        gpus_per_node=2,
        # 单GPU配置
        gpu_flops=142,  # TFLOPs
        gpu_mem_size=24,  # GB
        gpu_mem_bw=936,  # GB/s
        # 节点内通信配置（PCIe Gen4）
        intra_node_bw=16,  # GB/s (PCIe Gen4 x16双向)
        intra_node_latency=2,  # us
        # 节点间通信配置（InfiniBand）
        inter_node_bw=12.5,  # GB/s (100Gb/s = 12.5GB/s)
        inter_node_latency=5,  # us
        # 其他参数
        bits='bf16',
        compute_efficiency=0.6,
        memory_efficiency=0.6,
        comm_efficiency=0.6,
    ):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.total_gpus = num_nodes * gpus_per_node
        
        # GPU配置
        self.gpu_flops = gpu_flops
        self.gpu_mem_size = gpu_mem_size
        self.gpu_mem_bw = gpu_mem_bw
        
        # 节点内通信（PCIe）
        self.intra_node_bw = intra_node_bw
        self.intra_node_latency = intra_node_latency
        
        # 节点间通信（InfiniBand）
        self.inter_node_bw = inter_node_bw
        self.inter_node_latency = inter_node_latency
        
        self.bits = bits
        self.compute_efficiency = compute_efficiency
        self.memory_efficiency = memory_efficiency
        self.comm_efficiency = comm_efficiency
        
        # 创建节点内和节点间的System对象
        self._create_system_configs()
    
    def _create_system_configs(self):
        """创建节点内和节点间的System配置"""
        
        # 节点内System配置（用于节点内的张量并行通信）
        self.intra_node_system = System(
            flops=self.gpu_flops,
            mxu_shape=None,
            onchip_mem_bw=18000,
            on_chip_mem_size=float('Inf'),
            offchip_mem_bw=self.gpu_mem_bw,
            off_chip_mem_size=self.gpu_mem_size,
            external_mem_bw=0,
            frequency=940,
            bits=self.bits,
            compute_efficiency=self.compute_efficiency,
            memory_efficiency=self.memory_efficiency,
            comm_efficiency=self.comm_efficiency,
            interchip_link_bw=self.intra_node_bw,  # PCIe带宽
            num_nodes=self.gpus_per_node,
            interchip_link_latency=self.intra_node_latency,
            compute_engine='GenZ',
            collective_strategy='GenZ',
            topology='FullyConnected',
        )
        
        # 节点间System配置（用于节点间的流水线并行通信）
        self.inter_node_system = System(
            flops=self.gpu_flops * self.gpus_per_node,  # 一个节点的总算力
            mxu_shape=None,
            onchip_mem_bw=18000,
            on_chip_mem_size=float('Inf'),
            offchip_mem_bw=self.gpu_mem_bw,
            off_chip_mem_size=self.gpu_mem_size * self.gpus_per_node,
            external_mem_bw=0,
            frequency=940,
            bits=self.bits,
            compute_efficiency=self.compute_efficiency,
            memory_efficiency=self.memory_efficiency,
            comm_efficiency=self.comm_efficiency,
            interchip_link_bw=self.inter_node_bw,  # InfiniBand带宽
            num_nodes=self.num_nodes,
            interchip_link_latency=self.inter_node_latency,
            compute_engine='GenZ',
            collective_strategy='GenZ',
            topology='FullyConnected',
        )
    
    def get_intra_node_system(self):
        """获取节点内System配置（用于TP通信）"""
        return self.intra_node_system
    
    def get_inter_node_system(self):
        """获取节点间System配置（用于PP通信）"""
        return self.inter_node_system
    
    def __str__(self):
        info = f"=== 多节点系统配置 ===\n"
        info += f"节点数: {self.num_nodes}\n"
        info += f"每节点GPU数: {self.gpus_per_node}\n"
        info += f"总GPU数: {self.total_gpus}\n"
        info += f"\n--- 单GPU配置 ---\n"
        info += f"算力: {self.gpu_flops} TFLOPs\n"
        info += f"显存: {self.gpu_mem_size} GB\n"
        info += f"显存带宽: {self.gpu_mem_bw} GB/s\n"
        info += f"\n--- 节点内通信 (PCIe Gen4) ---\n"
        info += f"带宽: {self.intra_node_bw} GB/s\n"
        info += f"延迟: {self.intra_node_latency} us\n"
        info += f"\n--- 节点间通信 (InfiniBand) ---\n"
        info += f"带宽: {self.inter_node_bw} GB/s\n"
        info += f"延迟: {self.inter_node_latency} us\n"
        return info


def create_default_multi_node_system():
    """
    创建默认的多节点系统配置
    2节点 × 2 RTX3090
    """
    return MultiNodeSystem(
        num_nodes=2,
        gpus_per_node=2,
        gpu_flops=142,
        gpu_mem_size=24,
        gpu_mem_bw=936,
        intra_node_bw=16,
        intra_node_latency=2,
        inter_node_bw=12.5,
        inter_node_latency=5,
        bits='bf16',
        compute_efficiency=0.6,
        memory_efficiency=0.6,
        comm_efficiency=0.6,
    )
