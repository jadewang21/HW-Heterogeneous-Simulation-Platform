#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
import os
import time

from GenZ import decode_moddeling, prefill_moddeling, get_configs
from Systems.system_configs import system_configs 
from GenZ.system import System
from GenZ.collective_times import set_comm_backend

def run_batch_simulation_from_df(
    df,                        # DataFrame 包含 input_tokens / output_tokens
    model_to_simulate,
    quantization_setting,
    beam_size_value,
    batch_size_value,
    system_params,
    system_efficiency_value,
    tensor_parallel_nodes,
    pipeline_parallel_nodes,
    comm_backend,
    output_csv_filename
):
    warnings.filterwarnings("ignore")
    
    # 设置通信后端
    set_comm_backend(comm_backend)
    print(f"使用通信后端: {comm_backend}")
    
    # 验证必要的列
    required_columns = ['input_tokens', 'output_tokens']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV缺少必要的列: {col}")
    
    print(f"找到 {len(df)} 个请求，开始模拟: {output_csv_filename}")
    print(f"并行配置: TP={tensor_parallel_nodes}, PP={pipeline_parallel_nodes}")
    
    current_system_config = {}
    if isinstance(system_params, str):
        if system_params in system_configs:
            current_system_config = system_configs[system_params]
            print(f"使用系统配置: {system_params}")
        else:
            raise ValueError(f'指定的系统: {system_params} 不在预定义系统中。可用系统: {list(system_configs.keys())}')
    elif isinstance(system_params, dict):
        current_system_config = system_params
        print("使用自定义系统配置")
    else:
        raise ValueError('system_params 必须是系统名称字符串或自定义规格的字典。')

    results = []
    model_config = get_configs(model_to_simulate)
    actual_model_name = model_config.model
    
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
        simulation_time = np.nan
        notes = ""
        
        try:
            # 记录单次请求模拟开始时间
            start_time = time.time()
            batch_size = batch_size_value
            
            prefill_outputs = prefill_moddeling(
                model=model_to_simulate, 
                batch_size=batch_size,
                input_tokens=input_tokens,
                system_name=current_system_config, 
                system_eff=system_efficiency_value,
                bits=quantization_setting,
                tensor_parallel=tensor_parallel_nodes, 
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
                debug=False
            )
            tpot_ms = decode_outputs.get('Latency')
            decode_throughput = decode_outputs.get('Throughput')

            # E2E = TTFT + TPOT * 输出token数
            if ttft_ms is not None and tpot_ms is not None:
                e2e_ms = ttft_ms + tpot_ms * output_tokens
            
            # 记录单次请求模拟结束时间
            simulation_time = time.time() - start_time
            
        except Exception as e:
            notes = f"模拟失败: {e}"
            print(f"\n请求 {request_id} 模拟失败: {e}")
            # 即使失败也记录模拟时间
            simulation_time = time.time() - start_time
        
        results.append({
            'Request_ID': request_id,
            'Model': actual_model_name,
            'Batch': batch_size,
            'Input_Tokens': input_tokens,
            'Output_Tokens': output_tokens,
            'Beam_Size': beam_size_value,
            'TTFT(ms)': ttft_ms,
            'TPOT(ms)': tpot_ms,
            'E2E(ms)': e2e_ms,
            'Prefill_Throughput(tokens/s)': prefill_throughput,
            'Decode_Throughput(tokens/s)': decode_throughput,
            'Simulation_Time(s)': simulation_time,
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

    return results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GenZ batch simulation for chat/humaneval/mmlu CSVs")
    
    # 模型和量化参数
    parser.add_argument("--model", type=str, default="llama2_7b", help="Model to simulate")
    parser.add_argument("--quantization", type=str, default="bf16", help="Quantization setting")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for simulation")
    parser.add_argument("--system_eff", type=float, default=0.6, help="System efficiency")
    
    # 并行配置参数
    parser.add_argument("--tp_nodes", type=int, default=2, help="Tensor parallel nodes")
    parser.add_argument("--pp_nodes", type=int, default=1, help="Pipeline parallel nodes")
    
    # 系统配置参数
    parser.add_argument("--system", type=str, default="RTX3090_GPU", 
                        help="System configuration from system_configs.py")
    
    # 通信后端选择参数
    parser.add_argument("--comm_backend", type=str, default="pcie_like", 
                        choices=["nv_like", "pcie_like"],
                        help="通信后端类型: nv_like (NVIDIA-like) 或 pcie_like (PCIe-like)")

    parser.add_argument("--summary_dir", type=str,
                        default="../../benchmark-perf/3090-tp2-llama2-7b",
                        help="Directory containing chat.csv")

    args = parser.parse_args()
    
    # 显示可用系统配置
    if args.system not in system_configs:
        print("❌ 指定的系统配置不存在")
        print("可用系统配置:")
        for sys_name in system_configs.keys():
            print(f"  - {sys_name}")
        exit(1)

    
    for bench in ["chat"]:
        csv_path = os.path.join(args.summary_dir, f"{bench}.csv")
        if not os.path.exists(csv_path):
            print(f"❌ 缺少 {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        # 这里确保列名与模拟器要求一致
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
            batch_size_value=args.batch_size,
            system_params=args.system,
            system_efficiency_value=args.system_eff,
            tensor_parallel_nodes=args.tp_nodes,
            pipeline_parallel_nodes=args.pp_nodes,
            comm_backend=args.comm_backend,
            output_csv_filename=output_csv_name
        )
    
    print("\n✅ 所有基准测试模拟完成")
