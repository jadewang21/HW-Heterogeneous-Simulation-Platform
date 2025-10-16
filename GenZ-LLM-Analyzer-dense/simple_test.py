#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/wang/sim/delivery/GenZ-LLM-Analyzer-dense')

from GenZ.system import System
from GenZ.unit import Unit

# 创建系统
unit = Unit()
system = System(
    unit=unit,
    flops=142,
    l2_cache_size=40,
    l2_cache_bw=2000,
)

# 测试长序列
data_sz = unit.unit_to_raw(100, type='M')
hit_rate = system.get_l2_cache_hit_rate(
    data_sz=data_sz,
    op_type='Attend',
    access_pattern='mixed',
    phase='prefill',
    seq_len=2000
)
print(f"Hit rate: {hit_rate:.3f}")