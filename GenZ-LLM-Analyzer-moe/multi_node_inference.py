"""
多节点推理模块
支持TP=2, PP=2的混合并行推理建模
"""

import numpy as np
from GenZ import get_configs, create_full_prefill_model, create_full_decode_model
from GenZ.analyse_model import get_model_df
from GenZ.unit import Unit
from multi_node_system import MultiNodeSystem
from model_partitioning import ModelPartitionStrategy
from multi_node_collective import get_communication_time


class MultiNodeInference:
    """
    多节点推理类
    
    支持跨节点的分布式推理，正确区分节点内和节点间的通信开销
    """
    
    def __init__(
        self,
        model_name,
        multi_node_system,
        partition_strategy,
        batch_size=2,
        bits='bf16',
        system_eff=0.6,
    ):
        """
        Args:
            model_name: 模型名称（如 'llama2_7b'）
            multi_node_system: MultiNodeSystem对象
            partition_strategy: ModelPartitionStrategy对象
            batch_size: 批大小
            bits: 量化位数
            system_eff: 系统效率
        """
        self.model_name = model_name
        self.multi_node_system = multi_node_system
        self.partition_strategy = partition_strategy
        self.batch_size = batch_size
        self.bits = bits
        self.system_eff = system_eff
        
        # 获取模型配置
        self.model_config = get_configs(model_name)
        
        # 获取系统配置
        self.intra_node_system = multi_node_system.get_intra_node_system()
        self.inter_node_system = multi_node_system.get_inter_node_system()
        
    def estimate_prefill_time(self, input_tokens):
        """
        估计Prefill阶段的延迟
        
        Args:
            input_tokens: 输入token数量
            
        Returns:
            dict: 包含延迟和吞吐量信息
        """
        # 使用节点内系统配置进行张量并行建模
        # 注意：这里的tensor_parallel参数指的是节点内的TP
        model = create_full_prefill_model(
            name=self.model_config,
            input_sequence_length=input_tokens,
            tensor_parallel=self.partition_strategy.tensor_parallel
        )
        
        # 获取模型的DataFrame表示
        unit = Unit()
        model_df = get_model_df(
            model,
            system=self.intra_node_system,
            unit=unit,
            batch_size=self.batch_size,
            intermediate_on_chip=True
        )
        
        # 计算每层的通信开销并调整
        total_comm_time_adjustment = 0
        num_layers = self.model_config.num_decoder_layers
        
        for layer_idx in range(num_layers):
            partition_info = self.partition_strategy.get_partition_info(num_layers)
            layer_info = partition_info[layer_idx]
            
            # 获取该层的通信数据大小（简化估计）
            hidden_size = self.model_config.hidden_size
            data_size = self.batch_size * input_tokens * hidden_size * 2  # bf16=2 bytes
            
            # 节点内TP通信（All-Reduce）
            if layer_info['has_intra_node_comm']:
                intra_comm_time = get_communication_time(
                    'allreduce',
                    data_size,
                    True,
                    self.intra_node_system,
                    self.inter_node_system,
                    self.partition_strategy.tensor_parallel
                )
                # 注意：GenZ已经计算了通信时间，这里只是验证
            
            # 节点间PP通信（P2P）
            if layer_info['has_inter_node_comm']:
                inter_comm_time = get_communication_time(
                    'p2p',
                    data_size,
                    False,
                    self.intra_node_system,
                    self.inter_node_system
                )
                # 将节点间通信开销添加到总时间
                total_comm_time_adjustment += inter_comm_time
        
        # 计算总延迟
        unit = Unit()
        base_latency = model_df[f'Latency ({unit.unit_time})'].sum()
        total_latency = base_latency + total_comm_time_adjustment
        
        # 计算吞吐量
        throughput = (self.batch_size * input_tokens * 1000) / total_latency
        
        return {
            'Latency': total_latency,
            'Throughput': throughput,
            'Base_Latency': base_latency,
            'Inter_Node_Comm': total_comm_time_adjustment,
            'model_df': model_df
        }
    
    def estimate_decode_time(self, input_tokens, output_tokens):
        """
        估计Decode阶段的延迟
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            dict: 包含延迟和吞吐量信息
        """
        # 使用节点内系统配置进行张量并行建模
        model = create_full_decode_model(
            name=self.model_config,
            input_sequence_length=input_tokens,
            output_gen_tokens=output_tokens,
            tensor_parallel=self.partition_strategy.tensor_parallel
        )
        
        # 获取模型的DataFrame表示
        unit = Unit()
        model_df = get_model_df(
            model,
            system=self.intra_node_system,
            unit=unit,
            batch_size=self.batch_size,
            intermediate_on_chip=True
        )
        
        # 计算每层的通信开销并调整
        total_comm_time_adjustment = 0
        num_layers = self.model_config.num_decoder_layers
        
        for layer_idx in range(num_layers):
            partition_info = self.partition_strategy.get_partition_info(num_layers)
            layer_info = partition_info[layer_idx]
            
            # 获取该层的通信数据大小（简化估计）
            hidden_size = self.model_config.hidden_size
            data_size = self.batch_size * hidden_size * 2  # bf16=2 bytes, decode一次只生成1个token
            
            # 节点间PP通信（P2P）
            if layer_info['has_inter_node_comm']:
                inter_comm_time = get_communication_time(
                    'p2p',
                    data_size,
                    False,
                    self.intra_node_system,
                    self.inter_node_system
                )
                # 将节点间通信开销添加到总时间
                total_comm_time_adjustment += inter_comm_time
        
        # 计算每个token的平均延迟
        unit = Unit()
        base_latency = model_df[f'Latency ({unit.unit_time})'].sum()
        per_token_latency = base_latency + total_comm_time_adjustment
        
        # 计算吞吐量
        throughput = (self.batch_size * 1000) / per_token_latency
        
        return {
            'Latency': per_token_latency,
            'Throughput': throughput,
            'Base_Latency': base_latency,
            'Inter_Node_Comm': total_comm_time_adjustment,
            'model_df': model_df
        }
    
    def run_inference(self, input_tokens, output_tokens):
        """
        运行完整的推理模拟
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            
        Returns:
            dict: 完整的推理性能指标
        """
        # Prefill阶段
        prefill_result = self.estimate_prefill_time(input_tokens)
        
        # Decode阶段
        decode_result = self.estimate_decode_time(input_tokens, output_tokens)
        
        # 计算端到端延迟
        ttft = prefill_result['Latency']  # Time to First Token
        tpot = decode_result['Latency']  # Time per Output Token
        e2e_latency = ttft + tpot * output_tokens
        
        return {
            'TTFT(ms)': ttft,
            'TPOT(ms)': tpot,
            'E2E(ms)': e2e_latency,
            'Prefill_Throughput(tokens/s)': prefill_result['Throughput'],
            'Decode_Throughput(tokens/s)': decode_result['Throughput'],
            'Prefill_Inter_Node_Comm(ms)': prefill_result['Inter_Node_Comm'],
            'Decode_Inter_Node_Comm(ms)': decode_result['Inter_Node_Comm'],
        }
