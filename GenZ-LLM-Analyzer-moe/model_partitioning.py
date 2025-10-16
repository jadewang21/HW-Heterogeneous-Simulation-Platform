"""
模型分片策略模块
支持TP=2, PP=2的混合并行分片
"""

from GenZ.parallelism import ParallelismConfig


class ModelPartitionStrategy:
    """
    模型分片策略类
    
    定义如何在多节点系统上进行模型分片：
    - 节点内：张量并行 (TP=2)
    - 节点间：流水线并行 (PP=2)
    """
    
    def __init__(
        self,
        num_nodes=2,
        gpus_per_node=2,
        tensor_parallel=2,
        pipeline_parallel=2,
    ):
        """
        Args:
            num_nodes: 节点数量
            gpus_per_node: 每个节点的GPU数量
            tensor_parallel: 张量并行度（节点内）
            pipeline_parallel: 流水线并行度（节点间）
        """
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        
        # 验证配置
        assert gpus_per_node == tensor_parallel, \
            f"节点内GPU数({gpus_per_node})应等于张量并行度({tensor_parallel})"
        assert num_nodes == pipeline_parallel, \
            f"节点数({num_nodes})应等于流水线并行度({pipeline_parallel})"
        
        # 创建并行配置
        self.parallelism_config = ParallelismConfig(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            data_parallel=1,
            expert_parallel=1,
            sequence_parallel=1,
        )
    
    def get_stage_for_layer(self, layer_idx, total_layers):
        """
        确定某一层属于哪个流水线阶段（哪个节点）
        
        Args:
            layer_idx: 层索引 (0-based)
            total_layers: 总层数
            
        Returns:
            stage_id: 流水线阶段ID (0-based)
        """
        layers_per_stage = total_layers // self.pipeline_parallel
        stage_id = layer_idx // layers_per_stage
        
        # 确保最后几层也能分配到正确的阶段
        if stage_id >= self.pipeline_parallel:
            stage_id = self.pipeline_parallel - 1
            
        return stage_id
    
    def is_intra_node_comm(self, layer_idx, total_layers):
        """
        判断某层的通信是否为节点内通信
        
        对于张量并行，所有层都需要节点内的All-Reduce通信
        
        Args:
            layer_idx: 层索引
            total_layers: 总层数
            
        Returns:
            bool: True表示节点内通信，False表示节点间通信
        """
        # 张量并行的通信都是节点内的
        return True
    
    def is_inter_node_comm(self, layer_idx, total_layers):
        """
        判断某层是否需要节点间通信（流水线并行的点对点通信）
        
        Args:
            layer_idx: 层索引
            total_layers: 总层数
            
        Returns:
            bool: True表示需要节点间通信
        """
        layers_per_stage = total_layers // self.pipeline_parallel
        
        # 每个阶段的最后一层需要将激活值发送到下一个节点
        # （除了最后一个阶段）
        stage_id = self.get_stage_for_layer(layer_idx, total_layers)
        is_last_layer_in_stage = (layer_idx + 1) % layers_per_stage == 0
        is_not_last_stage = stage_id < self.pipeline_parallel - 1
        
        return is_last_layer_in_stage and is_not_last_stage
    
    def get_partition_info(self, total_layers):
        """
        获取完整的分片信息
        
        Returns:
            dict: 包含每一层的分片信息
        """
        partition_info = {}
        
        for layer_idx in range(total_layers):
            stage_id = self.get_stage_for_layer(layer_idx, total_layers)
            
            partition_info[layer_idx] = {
                'stage_id': stage_id,  # 流水线阶段（节点）
                'tp_group': list(range(self.tensor_parallel)),  # 张量并行组
                'has_intra_node_comm': self.is_intra_node_comm(layer_idx, total_layers),
                'has_inter_node_comm': self.is_inter_node_comm(layer_idx, total_layers),
            }
        
        return partition_info
    
    def print_partition_strategy(self, total_layers):
        """打印分片策略"""
        print("=== 模型分片策略 ===")
        print(f"总层数: {total_layers}")
        print(f"张量并行度 (TP): {self.tensor_parallel} (节点内)")
        print(f"流水线并行度 (PP): {self.pipeline_parallel} (节点间)")
        print(f"\n每个节点的层分配:")
        
        layers_per_stage = total_layers // self.pipeline_parallel
        for stage in range(self.pipeline_parallel):
            start_layer = stage * layers_per_stage
            if stage == self.pipeline_parallel - 1:
                end_layer = total_layers - 1
            else:
                end_layer = (stage + 1) * layers_per_stage - 1
            
            print(f"  节点 {stage}: 层 {start_layer} - {end_layer}")


def create_default_partition_strategy():
    """
    创建默认的分片策略
    TP=2 (节点内), PP=2 (节点间)
    """
    return ModelPartitionStrategy(
        num_nodes=2,
        gpus_per_node=2,
        tensor_parallel=2,
        pipeline_parallel=2,
    )
