#!/usr/bin/env python3
"""
测试debug信息是否正常工作
"""

import sys
import os
sys.path.insert(0, '/home/wang/sim/delivery/GenZ-LLM-Analyzer-dense')

from GenZ.system import System
from GenZ.unit import Unit

def test_debug_output():
    """测试debug输出"""
    print("🔧 测试debug输出")
    print("=" * 50)
    
    # 创建系统
    unit = Unit()
    system = System(
        unit=unit,
        flops=142,  # RTX 3090
        l2_cache_size=40,  # 40MB L2缓存
        l2_cache_bw=2000,  # 2TB/s L2带宽
    )
    
    # 测试长序列的缓存命中率
    print("测试长序列缓存命中率...")
    
    # 模拟长序列Attention算子
    data_sz = unit.unit_to_raw(100, type='M')  # 100MB数据
    hit_rate = system.get_l2_cache_hit_rate(
        data_sz=data_sz,
        op_type='Attend',
        access_pattern='mixed',
        phase='prefill',
        seq_len=2000
    )
    
    print(f"最终命中率: {hit_rate:.3f}")
    print("\n✅ Debug测试完成")

if __name__ == "__main__":
    test_debug_output()
