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
    df,                        
    model_to_simulate,
    quantization_setting,
    beam_size_value,
    system_params,
    system_efficiency_value,
    tensor_parallel_nodes,
    output_csv_filename
):
    warnings.filterwarnings("ignore")
    

    required_columns = ['input_tokens', 'output_tokens']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV缺少必要的列: {col}")
    
    print(f"找到 {len(df)} 个请求，开始模拟: {output_csv_filename}")
    
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
            batch_size = 2
            
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
            
        except Exception as e:
            notes = f"模拟失败: {e}"
            print(f"\n请求 {request_id} 模拟失败: {e}")
        
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
    
    parser.add_argument("--model", type=str, default="llama2_7b", help="Model to simulate")
    parser.add_argument("--quantization", type=str, default="bf16", help="Quantization setting")
    parser.add_argument("--beam_size", type=int, default=1, help="Beam size")
    parser.add_argument("--system_eff", type=float, default=0.6, help="System efficiency")
    parser.add_argument("--tp_nodes", type=int, default=2, help="Tensor parallel nodes")

    # RTX 3090 默认配置
    parser.add_argument("--flops", type=float, default=142, help="TFLOPs")
    parser.add_argument("--mem_size", type=float, default=24, help="Memory size (GB)")
    parser.add_argument("--mem_bw", type=float, default=936, help="Memory BW (GB/s)")
    parser.add_argument("--icn_bw", type=float, default=16, help="ICN BW (GB/s)")

    parser.add_argument("--summary_dir", type=str,
                        default="./test",
                        help="Directory containing chat.csv, humaneval.csv, mmlu.csv")

    args = parser.parse_args()
    
    rtx_3090_custom_config = {
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
            system_params=rtx_3090_custom_config,
            system_efficiency_value=args.system_eff,
            tensor_parallel_nodes=args.tp_nodes,
            output_csv_filename=output_csv_name
        )
    
    print("\n✅ 所有基准测试模拟完成")
