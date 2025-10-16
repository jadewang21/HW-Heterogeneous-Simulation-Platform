#!/usr/bin/env python3
"""
多节点LLaMA2推理模拟主程序

配置：
- 2个节点，每个节点2张RTX3090
- 节点内：PCIe Gen4互连，TP=2
- 节点间：InfiniBand 100Gb互连，PP=2
- 模型：LLaMA2-7B
"""

import numpy as np
import pandas as pd
from multi_node_system import create_default_multi_node_system
from model_partitioning import create_default_partition_strategy
from multi_node_inference import MultiNodeInference


def main():
    print("=" * 80)
    print("多节点分布式推理模拟 - LLaMA2-7B")
    print("=" * 80)
    
    # 创建多节点系统配置
    print("\n1. 创建多节点系统配置...")
    multi_node_system = create_default_multi_node_system()
    print(multi_node_system)
    
    # 创建模型分片策略
    print("\n2. 创建模型分片策略...")
    partition_strategy = create_default_partition_strategy()
    partition_strategy.print_partition_strategy(32)  # LLaMA2-7B有32层
    
    # 创建多节点推理引擎
    print("\n3. 初始化多节点推理引擎...")
    inference_engine = MultiNodeInference(
        model_name='llama2_7b',
        multi_node_system=multi_node_system,
        partition_strategy=partition_strategy,
        batch_size=2,
        bits='bf16',
        system_eff=0.6,
    )
    print("推理引擎初始化完成")
    
    # 定义测试场景
    test_cases = [
        {'input_tokens': 128, 'output_tokens': 128, 'name': '短序列'},
        {'input_tokens': 512, 'output_tokens': 512, 'name': '中等序列'},
        {'input_tokens': 1024, 'output_tokens': 1024, 'name': '长序列'},
        {'input_tokens': 2048, 'output_tokens': 512, 'name': '长输入短输出'},
    ]
    
    # 运行测试
    print("\n4. 运行推理模拟...")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试场景 {i}: {test_case['name']}")
        print(f"  输入tokens: {test_case['input_tokens']}")
        print(f"  输出tokens: {test_case['output_tokens']}")
        
        result = inference_engine.run_inference(
            input_tokens=test_case['input_tokens'],
            output_tokens=test_case['output_tokens']
        )
        
        print(f"\n  性能指标:")
        print(f"    TTFT (首token延迟):     {result['TTFT(ms)']:.2f} ms")
        print(f"    TPOT (每token延迟):      {result['TPOT(ms)']:.2f} ms")
        print(f"    E2E (端到端延迟):        {result['E2E(ms)']:.2f} ms")
        print(f"    Prefill吞吐量:           {result['Prefill_Throughput(tokens/s)']:.2f} tokens/s")
        print(f"    Decode吞吐量:            {result['Decode_Throughput(tokens/s)']:.2f} tokens/s")
        print(f"    Prefill节点间通信开销:   {result['Prefill_Inter_Node_Comm(ms)']:.2f} ms")
        print(f"    Decode节点间通信开销:    {result['Decode_Inter_Node_Comm(ms)']:.2f} ms")
        
        results.append({
            'Scenario': test_case['name'],
            'Input_Tokens': test_case['input_tokens'],
            'Output_Tokens': test_case['output_tokens'],
            **result
        })
    
    # 保存结果
    print("\n" + "=" * 80)
    print("5. 保存结果...")
    
    results_df = pd.DataFrame(results)
    output_file = 'multi_node_llama2_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"结果已保存到: {output_file}")
    
    # 打印汇总统计
    print("\n" + "=" * 80)
    print("6. 汇总统计")
    print("=" * 80)
    print(f"\n平均TTFT:                {results_df['TTFT(ms)'].mean():.2f} ms")
    print(f"平均TPOT:                {results_df['TPOT(ms)'].mean():.2f} ms")
    print(f"平均E2E:                 {results_df['E2E(ms)'].mean():.2f} ms")
    print(f"平均Prefill吞吐量:       {results_df['Prefill_Throughput(tokens/s)'].mean():.2f} tokens/s")
    print(f"平均Decode吞吐量:        {results_df['Decode_Throughput(tokens/s)'].mean():.2f} tokens/s")
    print(f"平均节点间通信开销比例:  {(results_df['Prefill_Inter_Node_Comm(ms)'].mean() / results_df['TTFT(ms)'].mean() * 100):.2f}%")
    
    print("\n" + "=" * 80)
    print("模拟完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
