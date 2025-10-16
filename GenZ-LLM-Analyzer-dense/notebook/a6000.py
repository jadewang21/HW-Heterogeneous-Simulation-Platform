#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
import os


from GenZ import decode_moddeling, prefill_moddeling, get_configs
from Systems.system_configs import system_configs 
from GenZ.system import System

def run_batch_simulation_from_df(
    df,                        # DataFrame 包含 input_tokens / output_tokens
    model_to_simulate,
    quantization_setting,
    beam_size_value,
    system_params,
    system_efficiency_value,
    tensor_parallel_nodes,
    pipeline_parallel_nodes,
    model_offload_enabled,
    output_csv_filename
):
    warnings.filterwarnings("ignore")
    
    # 验证必要的列
    required_columns = ['input_tokens', 'output_tokens']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV缺少必要的列: {col}")
    
    print(f"找到 {len(df)} 个请求，开始模拟: {output_csv_filename}")
    print(f"并行配置: TP={tensor_parallel_nodes}, PP={pipeline_parallel_nodes}")
    print(f"内存卸载: {'启用' if model_offload_enabled else '禁用'}")
    
    current_system_config = {}
    if isinstance(system_params, str):
        if system_params in system_configs:
            current_system_config = system_configs[system_params]
        else:
            raise ValueError(f'指定的系统: {system_params} 不在预定义系统中。')
    elif isinstance(system_params, dict):
        current_system_config = system_params
    else:
        raise ValueError('system_params 必须是系统名称字符串或自定义规格的字典。')

    results = []
    model_config = get_configs(model_to_simulate)
    actual_model_name = model_config.model
    
    # 构建并行层级结构字符串
    parallelism_heirarchy = f"TP{{{tensor_parallel_nodes}}}_EP{{1}}_PP{{{pipeline_parallel_nodes}}}"
    
    # 对每个请求进行模拟
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="模拟进度"):
        input_tokens = int(row['input_tokens'])
        output_tokens = int(row['output_tokens'])
        
        request_id = idx
        
        ttft_ms = np.nan
        tpot_ms = np.nan
        prefill_throughput = np.nan
        decode_throughput = np.nan
        e2e_ms = np.nan
        notes = ""
        
        try:
            batch_size = 1
            
            # 先尝试模型配置分析，检查内存需求
            try:
                prefill_outputs = prefill_moddeling(
                    model=model_to_simulate, 
                    batch_size=batch_size,
                    input_tokens=input_tokens,
                    system_name=current_system_config, 
                    system_eff=system_efficiency_value,
                    bits=quantization_setting,
                    tensor_parallel=tensor_parallel_nodes,
                    pipeline_parallel=pipeline_parallel_nodes,
                    parallelism_heirarchy=parallelism_heirarchy,
                    model_offload=model_offload_enabled,
                    debug=False
                )
                ttft_ms = prefill_outputs.get('Latency')
                prefill_throughput = prefill_outputs.get('Throughput')
                
                decode_outputs = decode_moddeling(
                    model=model_to_simulate, 
                    batch_size=batch_size, 
                    Bb=beam_size_value,
                    input_tokens=input_tokens, 
                    output_tokens=output_tokens,
                    system_name=current_system_config, 
                    system_eff=system_efficiency_value,
                    bits=quantization_setting,
                    tensor_parallel=tensor_parallel_nodes,
                    pipeline_parallel=pipeline_parallel_nodes,
                    parallelism_heirarchy=parallelism_heirarchy,
                    model_offload=model_offload_enabled,
                    debug=False
                )
                tpot_ms = decode_outputs.get('Latency')
                decode_throughput = decode_outputs.get('Throughput')

                # E2E = TTFT + TPOT * 输出token数
                if ttft_ms is not None and tpot_ms is not None:
                    e2e_ms = ttft_ms + tpot_ms * output_tokens
                    
            except ValueError as ve:
                # 如果内存不足，尝试启用模型卸载或增加并行度
                if "All params would not fit on chip" in str(ve) and not model_offload_enabled:
                    print(f"请求 {request_id}: 内存不足，尝试启用模型卸载...")
                    try:
                        prefill_outputs = prefill_moddeling(
                            model=model_to_simulate, 
                            batch_size=batch_size,
                            input_tokens=input_tokens,
                            system_name=current_system_config, 
                            system_eff=system_efficiency_value,
                            bits=quantization_setting,
                            tensor_parallel=tensor_parallel_nodes,
                            pipeline_parallel=pipeline_parallel_nodes,
                            parallelism_heirarchy=parallelism_heirarchy,
                            model_offload=True,  # 强制启用卸载
                            debug=False
                        )
                        ttft_ms = prefill_outputs.get('Latency')
                        prefill_throughput = prefill_outputs.get('Throughput')
                        
                        decode_outputs = decode_moddeling(
                            model=model_to_simulate, 
                            batch_size=batch_size, 
                            Bb=beam_size_value,
                            input_tokens=input_tokens, 
                            output_tokens=output_tokens,
                            system_name=current_system_config, 
                            system_eff=system_efficiency_value,
                            bits=quantization_setting,
                            tensor_parallel=tensor_parallel_nodes,
                            pipeline_parallel=pipeline_parallel_nodes,
                            parallelism_heirarchy=parallelism_heirarchy,
                            model_offload=True,  # 强制启用卸载
                            debug=False
                        )
                        tpot_ms = decode_outputs.get('Latency')
                        decode_throughput = decode_outputs.get('Throughput')

                        # E2E = TTFT + TPOT * 输出token数
                        if ttft_ms is not None and tpot_ms is not None:
                            e2e_ms = ttft_ms + tpot_ms * output_tokens
                        notes = "使用内存卸载"
                        
                    except Exception as e2:
                        notes = f"模拟失败（即使启用卸载）: {e2}"
                        print(f"请求 {request_id} 即使启用卸载仍失败: {e2}")
                else:
                    raise ve
            
        except Exception as e:
            notes = f"模拟失败: {e}"
            print(f"\n请求 {request_id} 模拟失败: {e}")
        
        results.append({
            'Request_ID': request_id,
            'Model': actual_model_name,
            'Batch': batch_size,
            'TP': tensor_parallel_nodes,
            'PP': pipeline_parallel_nodes,
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
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_filename, index=False)
    print(f"\n模拟结果已保存到: {output_csv_filename}")
    
    successful_sims = results_df['TTFT(ms)'].notna().sum()
    print(f"\n成功模拟: {successful_sims}/{len(results_df)} 个请求")
    
    if successful_sims > 0:
        print("\n统计信息:")
        print(f"平均 TTFT: {results_df['TTFT(ms)'].mean():.2f} ms")
        print(f"平均 TPOT: {results_df['TPOT(ms)'].mean():.2f} ms")
        print(f"平均 E2E:  {results_df['E2E(ms)'].mean():.2f} ms")
        
        # 显示内存卸载使用情况
        offload_count = results_df['Notes'].str.contains('卸载', na=False).sum()
        if offload_count > 0:
            print(f"使用内存卸载的请求: {offload_count}/{successful_sims}")

    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GenZ batch simulation for chat/humaneval/mmlu CSVs")
    #LLaMA2_70b
    parser.add_argument("--model", type=str, default="Llama-2-7B", help="Model to simulate")
    parser.add_argument("--quantization", type=str, default="bf16", help="Quantization setting (bf16, int8, int4)")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size")
    parser.add_argument("--system_eff", type=float, default=0.6, help="System efficiency")
    
    # 并行配置参数
    parser.add_argument("--tp_nodes", type=int, default=2, help="Tensor parallel nodes (推荐4个用于70B模型)")
    parser.add_argument("--pp_nodes", type=int, default=2, help="Pipeline parallel nodes")
    parser.add_argument("--enable_offload", action="store_true", help="Enable model offload to system memory")

    # A6000 硬件配置 (48GB显存)
    parser.add_argument("--flops", type=float, default=155, help="TFLOPs")
    parser.add_argument("--mem_size", type=float, default=48, help="Memory size (GB)")
    parser.add_argument("--mem_bw", type=float, default=768, help="Memory BW (GB/s)")
    parser.add_argument("--icn_bw", type=float, default=112.5, help="ICN BW (GB/s)")
    # parser.add_argument("--flops", type=float, default=142, help="TFLOPs")
    # parser.add_argument("--mem_size", type=float, default=24, help="Memory size (GB)")
    # parser.add_argument("--mem_bw", type=float, default=936, help="Memory BW (GB/s)")
    # parser.add_argument("--icn_bw", type=float, default=16, help="ICN BW (GB/s)")

    
    parser.add_argument("--summary_dir", type=str,
                        default="/home/wang/sim/bench/phy-test/direct_execute/a6000-lab/tp2-7b-01",
                        help="Directory containing chat.csv, humaneval.csv, mmlu.csv")

    args = parser.parse_args()
    
    # 输出配置信息
    print("=== 配置信息 ===")
    print(f"模型: {args.model}")
    print(f"量化: {args.quantization}")
    print(f"并行配置: TP={args.tp_nodes}, PP={args.pp_nodes}")
    print(f"内存卸载: {'启用' if args.enable_offload else '禁用'}")
    print(f"硬件: {args.mem_size}GB显存")
    
    # 检查配置合理性
    if args.model == "LLaMA2_70b" and args.tp_nodes < 2:
        print("⚠️ 警告: 70B模型建议使用至少2个GPU进行张量并行")
        print("   当前配置可能因显存不足而失败")
    
    # 创建A6000硬件配置
    a6000_config = {
        'Flops': args.flops,
        'Memory_size': args.mem_size,
        'Memory_BW': args.mem_bw,
        'ICN': args.icn_bw,
        'ICN_LL': 2,
        'real_values': True
    }

    # 循环处理chat/humaneval/mmlu
    for bench in ["chat", "humaneval", "mmlu"]:
        csv_path = os.path.join(args.summary_dir, f"{bench}.csv")
        if not os.path.exists(csv_path):
            print(f"❌ 缺少 {csv_path}")
            continue
        
        print(f"\n=== 处理 {bench.upper()} 基准测试 ===")
        df = pd.read_csv(csv_path)
        
        # 验证CSV文件格式
        if "input_tokens" not in df.columns:
            raise ValueError(f"{csv_path} 缺少 input_tokens 列")
        if "output_tokens" not in df.columns:
            raise ValueError(f"{csv_path} 缺少 output_tokens 列")

        output_csv_name = os.path.join(args.summary_dir, f"{bench}-sim.csv")
        
        run_batch_simulation_from_df(
            df=df,
            model_to_simulate=args.model,
            quantization_setting=args.quantization,
            beam_size_value=args.beam_size,
            system_params=a6000_config,
            system_efficiency_value=args.system_eff,
            tensor_parallel_nodes=args.tp_nodes,
            pipeline_parallel_nodes=args.pp_nodes,
            model_offload_enabled=args.enable_offload,
            output_csv_filename=output_csv_name
        )
    
    print("\n✅ 所有基准测试模拟完成")
    print("\n=== 运行建议 ===")
    print("如果遇到内存不足错误，可以尝试:")
    print("1. 增加张量并行度: --tp_nodes 4 或 8")
    print("2. 启用内存卸载: --enable_offload")
    print("3. 使用更低精度量化: --quantization int4")
    print("4. 使用流水线并行: --pp_nodes 2")
