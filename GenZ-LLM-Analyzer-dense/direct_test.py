#!/usr/bin/env python3
"""
直接测试operators模块
"""

import sys
import os
sys.path.append('/home/wang/sim/delivery/GenZ-LLM-Analyzer-att')

# 直接导入operators模块
import importlib.util
spec = importlib.util.spec_from_file_location("operators", "/home/wang/sim/delivery/GenZ-LLM-Analyzer-att/GenZ/operators.py")
operators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operators)

import numpy as np

def test_operators():
    """
    测试修正后的Logit和Attend算子
    """
    print("=== 测试修正后的算子 ===")
    
    # 测试参数
    B, H, M, N, D, Hkv = 1, 8, 2048, 2048, 64, 8
    
    print(f"测试参数: B={B}, H={H}, M={M}, N={N}, D={D}, Hkv={Hkv}")
    
    # 测试Logit算子
    print("\n1. 测试Logit算子:")
    logit_dim = ["Logit", B, H, M, N, D, Hkv, 4]  # 4是OpType.Logit
    logit_op = operators.Logit(dim=logit_dim, density=(1.0, 1.0, 1.0))
    
    original_flops = B * H * M * N * D
    corrected_flops = logit_op.get_num_ops()
    
    print(f"   原始FLOPs计算: {original_flops:,}")
    print(f"   修正FLOPs计算: {corrected_flops:,}")
    print(f"   包含softmax常数: {corrected_flops / original_flops:.2f}x")
    
    # 测试Attend算子
    print("\n2. 测试Attend算子:")
    attend_dim = ["Attend", B, H, M, N, D, Hkv, 5]  # 5是OpType.Attend
    attend_op = operators.Attend(dim=attend_dim, density=(1.0, 1.0, 1.0))
    
    original_flops = B * H * M * N * D
    corrected_flops = attend_op.get_num_ops()
    
    print(f"   原始FLOPs计算: {original_flops:,}")
    print(f"   修正FLOPs计算: {corrected_flops:,}")
    print(f"   包含softmax常数: {corrected_flops / original_flops:.2f}x")
    
    # 验证softmax常数因子
    expected_constant = 3.0
    actual_constant = corrected_flops / original_flops
    
    print(f"\n3. 验证softmax常数因子:")
    print(f"   期望常数: {expected_constant}")
    print(f"   实际常数: {actual_constant:.2f}")
    
    if abs(actual_constant - expected_constant) < 0.01:
        print("   ✓ softmax常数因子正确")
    else:
        print("   ✗ softmax常数因子不正确")
    
    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    test_operators()

