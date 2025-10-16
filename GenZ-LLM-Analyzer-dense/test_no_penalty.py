#!/usr/bin/env python3
"""
测试移除序列长度惩罚的效果
"""

import sys
import os
sys.path.insert(0, '/home/wang/sim/delivery/GenZ-LLM-Analyzer-dense')

from GenZ.system import System
from GenZ.unit import Unit

def test_data_size_penalty():
    """测试基于数据大小的惩罚效果"""
    print("🔧 测试基于数据大小的惩罚效果")
    print("=" * 60)
    
    # 创建系统
    unit = Unit()
    system = System(
        unit=unit,
        flops=142,  # RTX 3090
        l2_cache_size=6,  # 6MB L2缓存（修正后的容量）
        l2_cache_bw=2000,  # 2TB/s L2带宽
    )
    
    # 测试不同数据大小的缓存命中率
    test_cases = [
        # (data_size_MB, op_type, seq_len, description)
        (10, 'Attend', 1, "小数据量"),
        (30, 'Attend', 1, "中等数据量"),
        (60, 'Attend', 1, "大数据量"),
        (120, 'Attend', 1, "超大数据量"),
        (250, 'Attend', 1, "极大数据量"),
        (5, 'GEMM', 1, "小数据量GEMM"),
        (25, 'GEMM', 1, "中等数据量GEMM"),
        (60, 'GEMM', 1, "大数据量GEMM"),
        (150, 'GEMM', 1, "超大数据量GEMM"),
    ]
    
    print("数据大小 | 算子类型 | 命中率 | 描述")
    print("-" * 60)
    
    for data_size_mb, op_type, seq_len, description in test_cases:
        data_sz = unit.unit_to_raw(data_size_mb, type='M')
        
        # 测试prefill阶段
        hit_rate = system.get_l2_cache_hit_rate(
            data_sz=data_sz,
            op_type=op_type,
            access_pattern='mixed',
            phase='prefill',
            seq_len=seq_len
        )
        
        print(f"{data_size_mb:8.1f}MB | {op_type:8s} | {hit_rate:.3f} | {description}")
    
    print("\n✅ 基于数据大小惩罚测试完成")
    print("\n📊 预期结果：")
    print("- 大数据量时命中率被惩罚因子降低")
    print("- 超大数据量（200MB+）：惩罚20%")
    print("- 大数据量（100MB+）：惩罚40%")
    print("- 中等数据量（50MB+）：惩罚60%")
    print("- 小数据量（20MB+）：惩罚80%")
    print("- 6MB缓存容量下，大数据量命中率会显著降低")

if __name__ == "__main__":
    test_data_size_penalty()
