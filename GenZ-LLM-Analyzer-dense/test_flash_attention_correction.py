#!/usr/bin/env python3
"""
测试Flash Attention修正后的实现
验证FLOPs计算是否正确
"""

import sys
import os
sys.path.append('/home/wang/sim/delivery/GenZ-LLM-Analyzer-att')

from GenZ.Models.flash_attention_config import set_flash_attention_mode, is_corrected_flash_attention_enabled
from GenZ.Models import get_configs, create_full_prefill_model, create_full_decode_model
from GenZ.parallelism import ParallelismConfig
from GenZ.system import System
from GenZ.unit import Unit
from GenZ.analyse_model import get_model_df, get_summary_table
import pandas as pd

def test_flash_attention_correction():
    """
    测试Flash Attention修正后的实现
    """
    print("=== Flash Attention修正测试 ===")
    
    # 测试配置
    model_name = 'llama2_7b'
    input_sequence_length = 2048
    batch_size = 1
    
    # 创建系统配置
    system = System(
        frequency=1000,
        flops=2000,
        off_chip_mem_size=80*1024,
        compute_efficiency=0.8,
        memory_efficiency=0.8,
        bits='bf16'
    )
    
    # 创建并行配置
    parallelism_config = ParallelismConfig(
        tensor_parallel=1,
        expert_parallel=1,
        sequence_parallel=1,
        data_parallel=1
    )
    
    unit = Unit()
    
    print(f"测试模型: {model_name}")
    print(f"输入序列长度: {input_sequence_length}")
    print(f"批次大小: {batch_size}")
    print()
    
    # 测试1: 使用原始Flash Attention实现
    print("1. 测试原始Flash Attention实现...")
    set_flash_attention_mode(use_corrected=False)
    print(f"   修正模式: {is_corrected_flash_attention_enabled()}")
    
    try:
        # 创建prefill模型
        prefill_model_path = create_full_prefill_model(
            name=model_name,
            input_sequence_length=input_sequence_length,
            tensor_parallel=1,
            pipeline_parallel=1,
            expert_parallel=1
        )
        
        # 分析模型
        model_df_original = get_model_df(
            prefill_model_path, 
            system=system, 
            batch_size=batch_size, 
            intermediate_on_chip=True
        )
        
        summary_original = get_summary_table(model_df_original, unit)
        
        print(f"   原始实现 - 总FLOPs: {summary_original[f'MACs ({unit.unit_flop})'].values[0]:.2f}")
        print(f"   原始实现 - 注意力FLOPs: {get_attention_flops(model_df_original, unit):.2f}")
        
    except Exception as e:
        print(f"   原始实现测试失败: {e}")
        return False
    
    # 测试2: 使用修正后的Flash Attention实现
    print("\n2. 测试修正后的Flash Attention实现...")
    set_flash_attention_mode(use_corrected=True)
    print(f"   修正模式: {is_corrected_flash_attention_enabled()}")
    
    try:
        # 创建prefill模型
        prefill_model_path_corrected = create_full_prefill_model(
            name=model_name,
            input_sequence_length=input_sequence_length,
            tensor_parallel=1,
            pipeline_parallel=1,
            expert_parallel=1
        )
        
        # 分析模型
        model_df_corrected = get_model_df(
            prefill_model_path_corrected, 
            system=system, 
            batch_size=batch_size, 
            intermediate_on_chip=True
        )
        
        summary_corrected = get_summary_table(model_df_corrected, unit)
        
        print(f"   修正实现 - 总FLOPs: {summary_corrected[f'MACs ({unit.unit_flop})'].values[0]:.2f}")
        print(f"   修正实现 - 注意力FLOPs: {get_attention_flops(model_df_corrected, unit):.2f}")
        
    except Exception as e:
        print(f"   修正实现测试失败: {e}")
        return False
    
    # 比较结果
    print("\n3. 结果比较:")
    original_flops = summary_original[f'MACs ({unit.unit_flop})'].values[0]
    corrected_flops = summary_corrected[f'MACs ({unit.unit_flop})'].values[0]
    
    print(f"   原始实现总FLOPs: {original_flops:.2f}")
    print(f"   修正实现总FLOPs: {corrected_flops:.2f}")
    print(f"   FLOPs差异: {abs(corrected_flops - original_flops):.2f}")
    print(f"   差异百分比: {abs(corrected_flops - original_flops) / original_flops * 100:.2f}%")
    
    # 检查是否包含softmax常数因子
    print("\n4. 检查softmax常数因子:")
    original_attention_flops = get_attention_flops(model_df_original, unit)
    corrected_attention_flops = get_attention_flops(model_df_corrected, unit)
    
    print(f"   原始注意力FLOPs: {original_attention_flops:.2f}")
    print(f"   修正注意力FLOPs: {corrected_attention_flops:.2f}")
    
    if corrected_attention_flops > original_attention_flops:
        print("   ✓ 修正实现包含了softmax常数因子")
    else:
        print("   ✗ 修正实现可能没有正确包含softmax常数因子")
    
    print("\n=== 测试完成 ===")
    return True

def get_attention_flops(model_df, unit):
    """
    计算注意力层的FLOPs
    """
    attention_flops = 0
    for i in range(len(model_df)):
        if 'Logit' in model_df.loc[i, 'Op Type'] or 'Attend' in model_df.loc[i, 'Op Type']:
            attention_flops += model_df.loc[i, f'Num ops ({unit.unit_flop})']
    return attention_flops

if __name__ == "__main__":
    test_flash_attention_correction()

