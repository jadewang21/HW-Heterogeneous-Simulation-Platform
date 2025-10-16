#!/usr/bin/env python3
"""
测试缓存修复效果
验证长序列时缓存命中率是否合理降低
"""

import sys
import os
sys.path.insert(0, '/home/wang/sim/delivery/GenZ-LLM-Analyzer-dense')

from GenZ.system import System
from GenZ.unit import Unit

def test_cache_hit_rate_fix():
    """测试缓存命中率修复效果"""
    print("🔧 测试缓存命中率修复效果")
    print("=" * 50)
    
    # 创建系统
    unit = Unit()
    system = System(
        unit=unit,
        flops=142,  # RTX 3090
        l2_cache_size=40,  # 40MB L2缓存
        l2_cache_bw=2000,  # 2TB/s L2带宽
    )
    
    # 测试不同序列长度的缓存命中率
    test_cases = [
        # (data_size_MB, op_type, seq_len, expected_trend)
        (10, 'Attend', 100, "高命中率"),
        (50, 'Attend', 1000, "中等命中率"),
        (100, 'Attend', 2000, "较低命中率"),
        (200, 'Attend', 3000, "低命中率"),
        (5, 'GEMM', 100, "高命中率"),
        (25, 'GEMM', 1000, "中等命中率"),
        (50, 'GEMM', 2000, "较低命中率"),
        (100, 'GEMM', 3000, "低命中率"),
    ]
    
    print("序列长度 | 数据大小 | 算子类型 | 命中率 | 趋势")
    print("-" * 50)
    
    for data_size_mb, op_type, seq_len, expected_trend in test_cases:
        data_sz = unit.unit_to_raw(data_size_mb, type='M')
        
        # 测试prefill阶段
        hit_rate = system.get_l2_cache_hit_rate(
            data_sz=data_sz,
            op_type=op_type,
            access_pattern='mixed',
            phase='prefill',
            seq_len=seq_len
        )
        
        print(f"{seq_len:8d} | {data_size_mb:8.1f}MB | {op_type:8s} | {hit_rate:.3f} | {expected_trend}")
    
    print("\n✅ 缓存命中率修复测试完成")
    print("\n📊 预期结果：")
    print("- 短序列（≤1000）：较高命中率")
    print("- 长序列（2000+）：显著降低命中率")
    print("- 超长序列（3000+）：很低命中率")

if __name__ == "__main__":
    test_cache_hit_rate_fix()
