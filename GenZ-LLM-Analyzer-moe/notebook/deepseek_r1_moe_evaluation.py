#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1 MoE 模型性能评估脚本

参考 a6000.py 的评估方式，专门针对 MoE 模型的性能评估
支持用户灵活指定硬件配置，计算 TTFT、TPOT、E2E 和吞吐量，保存到 CSV
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import warnings
import argparse
from tqdm import tqdm

# 导入GenZ核心模块  
import sys
sys.path.append('/home/wang/sim/org-genz/GenZ-LLM-Analyzer')

from GenZ import (
    decode_moddeling, 
    prefill_moddeling,
    ModelConfig,
    System
)
from GenZ.Models.default_models import MODEL_DICT
from Systems.system_configs import system_configs

class DeepSeekR1MoEEvaluator:
    """DeepSeek-R1 MoE模型性能评估器"""
    
    def __init__(self, config_path="ds-r1-config.json"):
        """初始化评估器，加载真实配置"""
        
        # 读取DeepSeek-R1配置
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config_json = json.load(f)
        else:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        # 创建并注册模型
        self._create_and_register_model()
        
        print("✓ DeepSeek-R1 MoE模型配置已加载")
        print(f"  - 层数: {self.config_json['num_hidden_layers']}")
        print(f"  - 隐藏维度: {self.config_json['hidden_size']} ")
        print(f"  - MoE专家数: {self.config_json['n_routed_experts']}")
        print(f"  - 激活专家数: {self.config_json['num_experts_per_tok']}")
        print(f"  - 词汇表: {self.config_json['vocab_size']}")
    
    def _create_and_register_model(self):
        """创建并注册模型配置"""
        config = self.config_json
        
        model_config = ModelConfig(
            model='deepseek-r1',
            vocab_size=config['vocab_size'],
            max_model_len=config['max_position_embeddings'], 
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            num_decoder_layers=config['num_hidden_layers'],
            num_attention_heads=config['num_attention_heads'],
            num_key_value_heads=config['num_key_value_heads'],
            head_dim=config['v_head_dim'],
            hidden_act=config['hidden_act'],
            
            # MoE 核心配置
            num_experts=config['n_routed_experts'],
            expert_top_k=config['num_experts_per_tok'],
            moe_intermediate_size=config['moe_intermediate_size'],
            n_shared_experts=config['n_shared_experts'],
            shared_expert_intermediate_size=config['intermediate_size'],
            first_k_dense_replace=config['first_k_dense_replace'],
            moe_layer_freq=config['moe_layer_freq']
        )
        
        try:
            MODEL_DICT.add_model(model_config)
            print("✓ 模型已注册到GenZ系统")
        except Exception as e:
            print(f"模型注册警告: {e}")
    
    def run_batch_simulation_from_df(self,
                                    df,
                                    hardware_config,
                                    quantization_setting='fp8',
                                    beam_size_value=1,
                                    system_efficiency_value=0.8,
                                    tensor_parallel_nodes=8,
                                    pipeline_parallel_nodes=1,
                                    model_offload_enabled=False,
                                    output_csv_filename='deepseek_r1_results.csv',
                                    # 多节点配置参数
                                    num_nodes=1,
                                    interchip_link_bw=450,
                                    interchip_link_latency=1.9,
                                    topology='FullyConnected'):
        """
        批量模拟评估，参考 a6000.py 的实现方式
        专门针对 MoE 模型进行优化
        """
        
        warnings.filterwarnings("ignore")
        
        # 验证必要列
        required_columns = ['input_tokens', 'output_tokens']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV缺少必要的列: {col}")
        
        print(f"找到 {len(df)} 个请求，开始 DeepSeek-R1 MoE 模拟")
        print(f"并行配置: TP={tensor_parallel_nodes}, PP={pipeline_parallel_nodes}")
        print(f"多节点配置: {num_nodes}个节点, 拓扑={topology}")
        print(f"节点互连: {interchip_link_bw}GB/s带宽, {interchip_link_latency}μs延迟")
        print(f"量化设置: {quantization_setting}")
        print(f"内存卸载: {'启用' if model_offload_enabled else '禁用'}")
        
        # 关键修复：正确构建GenZ多节点系统对象
        # 4个节点，每个节点8张卡，总共32张卡
        cards_per_node = 8
        total_cards = num_nodes * cards_per_node
        
        # 构建GenZ System对象，包含完整的多节点配置
        print(f"🔧 构建多节点GenZ系统...")
        print(f"   配置：{num_nodes}节点 × {cards_per_node}卡/节点 = {total_cards}张卡")
        print(f"   节点互连：{interchip_link_bw} GB/s, 延迟：{interchip_link_latency} μs")
        print(f"   拓扑结构：{topology}")
        
        # 关键修复：正确构建GenZ多节点系统配置
        # 首先添加RTX 4090配置到系统配置字典
        if 'RTX_4090_GPU' not in system_configs:
            system_configs['RTX_4090_GPU'] = {
                'Flops': hardware_config['Flops'], 
                'Memory_size': hardware_config['Memory_size'], 
                'Memory_BW': hardware_config['Memory_BW'], 
                'ICN': hardware_config.get('ICN', 16), 
                'real_values': True
            }
        
        # 正确构建多节点System对象
        system = System(
            flops=hardware_config['Flops'],  # 单卡算力，GenZ会自动处理多节点聚合
            off_chip_mem_size=hardware_config['Memory_size'] * 1024,  # 单卡内存(MB)
            offchip_mem_bw=hardware_config['Memory_BW'],  # 单卡带宽
            interchip_link_bw=interchip_link_bw,  # 节点间互连带宽
            interchip_link_latency=interchip_link_latency,  # 节点间延迟
            bits=quantization_setting,
            compute_efficiency=system_efficiency_value,
            memory_efficiency=system_efficiency_value,
            comm_efficiency=system_efficiency_value,
            num_nodes=num_nodes,  # GenZ会根据此参数自动计算总资源
            topology=topology,
            collective_strategy='GenZ',
            parallelism_heirarchy=f"TP{{{tensor_parallel_nodes}}}_EP{{1}}_PP{{{pipeline_parallel_nodes}}}"
        )
        
        print(f"✅ 系统构建完成：")
        print(f"   单卡算力：{system.unit.raw_to_unit(system.flops, type='C')} TFLOPs")
        print(f"   单卡内存：{system.unit.raw_to_unit(system.off_chip_mem_size, type='M')/1024:.1f} GB")
        print(f"   集群总算力：{system.unit.raw_to_unit(system.flops, type='C') * total_cards} TFLOPs")
        print(f"   集群总内存：{system.unit.raw_to_unit(system.off_chip_mem_size, type='M')/1024 * total_cards:.1f} GB")
        print(f"   节点间带宽：{system.unit.raw_to_unit(system.interchip_link_bw, type='BW')} GB/s")
        
        results = []
        
        # 构建并行层级结构字符串 (参考 a6000.py)
        parallelism_heirarchy = f"TP{{{tensor_parallel_nodes}}}_EP{{1}}_PP{{{pipeline_parallel_nodes}}}"
        
        # 逐个请求进行评估
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="MoE模拟进度"):
            input_tokens = int(row['input_tokens'])
            output_tokens = int(row['output_tokens'])
            request_id = idx
            batch_size = 1  # MoE模型通常使用较小的batch size
            
            # 初始化结果
            ttft_ms = np.nan
            tpot_ms = np.nan
            e2e_ms = np.nan
            prefill_throughput = np.nan
            decode_throughput = np.nan
            notes = ""
            
            try:
                # 1. 预填充阶段评估
                try:
                    prefill_outputs = prefill_moddeling(
                        model='deepseek-r1',
                        batch_size=batch_size,
                        input_tokens=input_tokens,
                        system_name=system,  # 使用构建的System对象
                        system_eff=system_efficiency_value,
                        bits=quantization_setting,
                        tensor_parallel=tensor_parallel_nodes,
                        pipeline_parallel=pipeline_parallel_nodes,
                        parallelism_heirarchy=parallelism_heirarchy,
                        model_offload=model_offload_enabled,
                        debug=False
                    )
                    
                    ttft_ms = prefill_outputs.get('Latency', np.nan)
                    prefill_throughput = prefill_outputs.get('Throughput', np.nan)
                    
                except Exception as prefill_error:
                    print(f"请求 {request_id}: 预填充失败 - {prefill_error}")
                    ttft_ms = np.nan
                    prefill_throughput = np.nan
                
                # 2. 解码阶段评估
                try:
                    decode_outputs = decode_moddeling(
                        model='deepseek-r1',
                        batch_size=batch_size,
                        Bb=beam_size_value,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        system_name=system,  # 使用构建的System对象
                        system_eff=system_efficiency_value,
                        bits=quantization_setting,
                        tensor_parallel=tensor_parallel_nodes,
                        pipeline_parallel=pipeline_parallel_nodes,
                        parallelism_heirarchy=parallelism_heirarchy,
                        model_offload=model_offload_enabled,
                        debug=False
                    )
                    
                    decode_latency = decode_outputs.get('Latency', np.nan)
                    decode_throughput = decode_outputs.get('Throughput', np.nan)
                    
                    if not np.isnan(decode_latency) and output_tokens > 0:
                        tpot_ms = decode_latency / output_tokens
                    else:
                        tpot_ms = np.nan
                        
                except Exception as decode_error:
                    print(f"请求 {request_id}: 解码失败 - {decode_error}")
                    tpot_ms = np.nan
                    decode_throughput = np.nan
                
                # 3. 计算端到端指标
                if not np.isnan(ttft_ms) and not np.isnan(tpot_ms):
                    e2e_ms = ttft_ms + (tpot_ms * output_tokens)
                else:
                    e2e_ms = np.nan
                    
                # 处理内存不足的情况 (参考 a6000.py 的错误处理)
            except ValueError as ve:
                if "All params would not fit on chip" in str(ve) and not model_offload_enabled:
                    print(f"请求 {request_id}: 内存不足，尝试启用模型卸载...")
                    notes = "内存不足，需要卸载或更多GPU"
                else:
                    notes = f"评估失败: {ve}"
                    print(f"请求 {request_id} 失败: {ve}")
            except Exception as e:
                notes = f"评估失败: {e}"
                print(f"请求 {request_id} 意外错误: {e}")
            
            # 保存结果，包含多节点信息
            results.append({
                'Request_ID': request_id,
                'Model': 'deepseek-r1',
                'Batch': batch_size,
                'TP': tensor_parallel_nodes,
                'PP': pipeline_parallel_nodes,
                'Num_Nodes': num_nodes,
                'Node_Topology': topology,
                'Interchip_BW(GB/s)': interchip_link_bw,
                'Interchip_Latency(us)': interchip_link_latency,
                'Input_Tokens': input_tokens,
                'Output_Tokens': output_tokens,
                'Beam_Size': beam_size_value,
                'TTFT(ms)': ttft_ms,
                'TPOT(ms)': tpot_ms,
                'E2E(ms)': e2e_ms,
                'Prefill_Throughput(tokens/s)': prefill_throughput,
                'Decode_Throughput(tokens/s)': decode_throughput,
                'Notes': notes
            })
        
        # 保存到CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_filename, index=False)
        print(f"\n✅ MoE模拟结果已保存到: {output_csv_filename}")
        
        # 统计分析
        successful_sims = results_df['TTFT(ms)'].notna().sum()
        print(f"成功模拟: {successful_sims}/{len(results_df)} 个请求")
        
        if successful_sims > 0:
            print("\n=== MoE模型性能统计 ===")
            print(f"平均 TTFT: {results_df['TTFT(ms)'].mean():.2f} ms")
            print(f"平均 TPOT: {results_df['TPOT(ms)'].mean():.2f} ms") 
            print(f"平均 E2E:  {results_df['E2E(ms)'].mean():.2f} ms")
            
            # MoE特有分析
            if not results_df['Prefill_Throughput(tokens/s)'].isna().all():
                print(f"预填充吞吐量: {results_df['Prefill_Throughput(tokens/s)'].mean():.2f} tokens/s")
            if not results_df['Decode_Throughput(tokens/s)'].isna().all():
                print(f"解码吞吐量: {results_df['Decode_Throughput(tokens/s)'].mean():.2f} tokens/s")
        
        return results_df
    
    def create_test_workload(self, workload_type='mixed'):
        """创建测试工作负载"""
        
        if workload_type == 'chat':
            # 对话场景
            data = [
                {'input_tokens': 1024, 'output_tokens': 256},
                {'input_tokens': 2048, 'output_tokens': 512},
                {'input_tokens': 4096, 'output_tokens': 1024},
            ]
        elif workload_type == 'code':
            # 代码生成场景
            data = [
                {'input_tokens': 2048, 'output_tokens': 512},
                {'input_tokens': 4096, 'output_tokens': 1024},
                {'input_tokens': 8192, 'output_tokens': 2048},
            ]
        elif workload_type == 'long_context':
            # 长文本场景
            data = [
                {'input_tokens': 16384, 'output_tokens': 1024},
                {'input_tokens': 32768, 'output_tokens': 2048},
                {'input_tokens': 65536, 'output_tokens': 4096},
            ]
        else:  # mixed
            # 混合场景
            data = [
                {'input_tokens': 1024, 'output_tokens': 256},
                {'input_tokens': 2048, 'output_tokens': 512},
                {'input_tokens': 4096, 'output_tokens': 1024},
                {'input_tokens': 8192, 'output_tokens': 1024},
                {'input_tokens': 16384, 'output_tokens': 2048},
            ]
        
        df = pd.DataFrame(data)
        return df


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="DeepSeek-R1 MoE 模型性能评估")
    
    # 硬件配置参数 (RTX 4090配置，针对32卡集群优化)
    parser.add_argument("--flops", type=float, default=330, help="TFLOPs (RTX 4090: ~330)")
    parser.add_argument("--mem_size", type=float, default=24, help="Memory size (GB, RTX 4090: 24GB)")
    parser.add_argument("--mem_bw", type=float, default=1008, help="Memory BW (GB/s, RTX 4090: ~1008)")
    parser.add_argument("--icn_bw", type=float, default=16, help="ICN BW (GB/s, NVLink/PCIe)")
    
    # 评估配置
    parser.add_argument("--quantization", type=str, default="bf16", 
                       choices=["fp32", "f32", "fp16", "bf16", "int8", "int4", "fp8"],
                       help="量化设置")
    parser.add_argument("--beam_size", type=int, default=1, help="束搜索大小")
    parser.add_argument("--system_eff", type=float, default=0.8, help="系统效率")
    
    # 并行配置
    parser.add_argument("--tp_nodes", type=int, default=8, 
                       help="张量并行节点数 (MoE模型推荐8+)")
    parser.add_argument("--pp_nodes", type=int, default=4, help="流水线并行节点数")
    parser.add_argument("--enable_offload", action="store_true", help="启用模型卸载")
    
    # 多节点配置参数
    parser.add_argument("--num_nodes", type=int, default=4, help="节点数量")
    parser.add_argument("--interchip_bw", type=float, default=10, 
                       help="节点间互连带宽 (GB/s)")
    parser.add_argument("--interchip_latency", type=float, default=1.9, 
                       help="节点间通信延迟 (μs)")
    parser.add_argument("--topology", type=str, default="FullyConnected",
                       choices=["FullyConnected", "Ring", "2DTorus"], 
                       help="网络拓扑结构")
    
    # 工作负载
    parser.add_argument("--workload", type=str, default="mixed",
                       choices=["chat", "code", "long_context", "mixed"],
                       help="测试工作负载类型")
    parser.add_argument("--input_csv", type=str, default="/home/wang/sim/bench/deepseek/chat.csv",
                       help="输入CSV文件路径(可选，包含input_tokens和output_tokens列)")
    
    # 输出
    parser.add_argument("--output_csv", type=str, default="deepseek_r1_moe_results.csv",
                       help="输出CSV文件路径")
    
    args = parser.parse_args()
    
    print("DeepSeek-R1 MoE 模型性能评估工具")
    print("="*50)
    
    try:
        # 初始化评估器
        config_path = os.path.join(os.path.dirname(__file__), 'ds-r1-config.json')
        evaluator = DeepSeekR1MoEEvaluator(config_path)
        
        # 创建硬件配置
        hardware_config = {
            'Flops': args.flops,
            'Memory_size': args.mem_size,
            'Memory_BW': args.mem_bw,
            'ICN': args.icn_bw,
            'real_values': True
        }
        
        print(f"\n=== 硬件配置 (单节点) ===")
        print(f"算力: {args.flops} TFLOPs")
        print(f"显存: {args.mem_size} GB") 
        print(f"内存带宽: {args.mem_bw} GB/s")
        print(f"互连带宽: {args.icn_bw} GB/s")
        
        print(f"\n=== 集群总配置 ({args.num_nodes}节点) ===")
        print(f"总算力: {args.flops * args.num_nodes} TFLOPs")
        print(f"总内存: {args.mem_size * args.num_nodes} GB")
        
        print(f"\n=== 评估配置 ===")
        print(f"量化: {args.quantization}")
        print(f"并行配置: TP={args.tp_nodes}, PP={args.pp_nodes}")
        print(f"系统效率: {args.system_eff}")
        
        print(f"\n=== 多节点配置 ===")
        print(f"节点数量: {args.num_nodes}")
        print(f"节点拓扑: {args.topology}")
        print(f"互连带宽: {args.interchip_bw} GB/s")
        print(f"互连延迟: {args.interchip_latency} μs")
        
        # 准备测试数据
        if args.input_csv and os.path.exists(args.input_csv):
            print(f"\n使用自定义CSV: {args.input_csv}")
            df = pd.read_csv(args.input_csv)
        else:
            print(f"\n使用预设工作负载: {args.workload}")
            df = evaluator.create_test_workload(args.workload)
        
        print(f"测试请求数: {len(df)}")
        
        # 运行评估
        print(f"\n开始 DeepSeek-R1 MoE 性能评估...")
        results_df = evaluator.run_batch_simulation_from_df(
            df=df,
            hardware_config=hardware_config,
            quantization_setting=args.quantization,
            beam_size_value=args.beam_size,
            system_efficiency_value=args.system_eff,
            tensor_parallel_nodes=args.tp_nodes,
            pipeline_parallel_nodes=args.pp_nodes,
            model_offload_enabled=args.enable_offload,
            output_csv_filename=args.output_csv,
            # 多节点参数
            num_nodes=args.num_nodes,
            interchip_link_bw=args.interchip_bw,
            interchip_link_latency=args.interchip_latency,
            topology=args.topology
        )
        
        print(f"\n✅ 评估完成，结果保存至: {args.output_csv}")
        
        print("\n=== MoE模型特性提示 ===")
        print("1. DeepSeek-R1 使用 MoE 架构，实际计算量比参数量小")
        print("2. 建议使用 FP8 量化以节省内存")
        print("3. 张量并行度建议 8+ 以充分利用专家并行性")
        print("4. 长序列场景下注意 KV 缓存内存占用")
        
    except FileNotFoundError as e:
        print(f"❌ 配置文件错误: {e}")
        print("请确保 ds-r1-config.json 文件存在")
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        print("请检查配置参数和环境")


if __name__ == "__main__":
    main()
