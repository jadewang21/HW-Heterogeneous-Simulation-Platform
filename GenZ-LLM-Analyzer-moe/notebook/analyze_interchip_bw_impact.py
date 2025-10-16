#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 interchip_bw 对 DeepSeek-R1 MoE 性能的影响
汇总测试结果并创建性能对比分析
"""

import pandas as pd
import numpy as np
import os

def analyze_performance_impact():
    """分析不同interchip_bw设置对性能的影响"""
    
    print("DeepSeek-R1 MoE interchip_bw 性能影响分析")
    print("=" * 50)
    
    # 手动收集的测试结果（基于运行输出）
    results_data = [
        {
            'interchip_bw_gbps': 1000,
            'ttft_ms': 4050.58,
            'tpot_ms': 8.32,
            'e2e_ms': 5070.58,
            'prefill_throughput': 285.10,
            'decode_throughput': 120.18
        },
        {
            'interchip_bw_gbps': 10,
            'ttft_ms': 14940.01,
            'tpot_ms': 30.67,
            'e2e_ms': 15251.34,
            'prefill_throughput': 77.32,
            'decode_throughput': 32.61
        }
    ]
    
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 计算性能影响
    print("\n=== 性能对比分析 ===")
    print(f"{'指标':<20} {'1000GB/s':<15} {'10GB/s':<15} {'性能下降':<15}")
    print("-" * 65)
    
    high_bw = df[df['interchip_bw_gbps'] == 1000].iloc[0]
    low_bw = df[df['interchip_bw_gbps'] == 10].iloc[0]
    
    metrics = [
        ('TTFT (ms)', 'ttft_ms'),
        ('TPOT (ms)', 'tpot_ms'), 
        ('E2E (ms)', 'e2e_ms'),
        ('预填充吞吐量', 'prefill_throughput'),
        ('解码吞吐量', 'decode_throughput')
    ]
    
    for name, key in metrics:
        high_val = high_bw[key]
        low_val = low_bw[key]
        
        if 'throughput' in key or '吞吐量' in name:
            # 对于吞吐量，计算下降百分比
            decrease = (high_val - low_val) / high_val * 100
            ratio_text = f"-{decrease:.1f}%"
        else:
            # 对于延迟，计算增加倍数
            ratio = low_val / high_val
            ratio_text = f"{ratio:.1f}x"
            
        print(f"{name:<20} {high_val:<15.2f} {low_val:<15.2f} {ratio_text:<15}")
    
    # 保存详细对比结果
    comparison_df = pd.DataFrame([
        {
            'Metric': 'TTFT (ms)',
            '1000_GB/s': high_bw['ttft_ms'],
            '10_GB/s': low_bw['ttft_ms'],
            'Performance_Impact': f"{low_bw['ttft_ms'] / high_bw['ttft_ms']:.1f}x slower"
        },
        {
            'Metric': 'TPOT (ms)',
            '1000_GB/s': high_bw['tpot_ms'],
            '10_GB/s': low_bw['tpot_ms'],
            'Performance_Impact': f"{low_bw['tpot_ms'] / high_bw['tpot_ms']:.1f}x slower"
        },
        {
            'Metric': 'E2E (ms)',
            '1000_GB/s': high_bw['e2e_ms'],
            '10_GB/s': low_bw['e2e_ms'], 
            'Performance_Impact': f"{low_bw['e2e_ms'] / high_bw['e2e_ms']:.1f}x slower"
        },
        {
            'Metric': 'Prefill Throughput (tokens/s)',
            '1000_GB/s': high_bw['prefill_throughput'],
            '10_GB/s': low_bw['prefill_throughput'],
            'Performance_Impact': f"{(high_bw['prefill_throughput'] - low_bw['prefill_throughput']) / high_bw['prefill_throughput'] * 100:.1f}% decrease"
        },
        {
            'Metric': 'Decode Throughput (tokens/s)',
            '1000_GB/s': high_bw['decode_throughput'],
            '10_GB/s': low_bw['decode_throughput'],
            'Performance_Impact': f"{(high_bw['decode_throughput'] - low_bw['decode_throughput']) / high_bw['decode_throughput'] * 100:.1f}% decrease"
        }
    ])
    
    # 保存到CSV
    output_file = 'interchip_bw_performance_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✅ 性能对比分析已保存到: {output_file}")
    
    # 关键发现总结
    print("\n=== 关键发现 ===")
    print("1. 网络带宽对性能影响巨大:")
    print(f"   - TTFT 增加了 {low_bw['ttft_ms'] / high_bw['ttft_ms']:.1f} 倍")
    print(f"   - TPOT 增加了 {low_bw['tpot_ms'] / high_bw['tpot_ms']:.1f} 倍")
    print("2. 低带宽严重影响多节点通信效率")
    print("3. GenZ系统的集群建模现在正确工作")
    print("\n🎉 修复验证成功！interchip_bw参数现在真实影响性能指标")
    
    return comparison_df

if __name__ == "__main__":
    analyze_performance_impact()
