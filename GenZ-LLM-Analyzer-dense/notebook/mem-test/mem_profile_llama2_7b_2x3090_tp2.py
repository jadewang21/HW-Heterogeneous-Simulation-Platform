#!/usr/bin/env python3
import os
import warnings
import pandas as pd

from GenZ import prefill_moddeling, decode_moddeling


def main():
    warnings.filterwarnings("ignore")

    # Target: 2x RTX 3090, tensor parallel = 1
    system_3090 = {
        'Flops': 142,           # TFLOPs per GPU
        'Memory_size': 24,      # GB per GPU
        'Memory_BW': 936,       # GB/s
        'ICN': 16,              # GB/s interconnect
        'ICN_LL': 2,            # us latency (approx)
        'real_values': True,
    }

    model_name = 'llama2_7b'
    tensor_parallel = 1

    # Example workload
    batch_size = 1
    input_tokens = 4096
    output_tokens = 128
    bits = 'bf16'

    # Profile Prefill (profiling mode returns (model_df, summary_table))
    prefill_df, prefill_summary = prefill_moddeling(
        model=model_name,
        batch_size=batch_size,
        input_tokens=input_tokens,
        system_name=system_3090,
        bits=bits,
        tensor_parallel=tensor_parallel,
        model_profilling=True,
    )

    # Profile Decode (profiling mode returns (model_df, summary_table))
    decode_df, decode_summary = decode_moddeling(
        model=model_name,
        batch_size=batch_size,
        Bb=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        system_name=system_3090,
        bits=bits,
        tensor_parallel=tensor_parallel,
        model_profilling=True,
    )

    # Expected columns are in MB per device
    cols = [
        'Total Weights (MB)',
        'KV Cache (MB)',
        'Unused Weights (MB)',
        'On-chip Memory Footprint (MB)'
    ]

    # Some models may not populate Unused Weights; ensure column exists
    for df in (prefill_summary, decode_summary):
        if 'Unused Weights (MB)' not in df.columns:
            df['Unused Weights (MB)'] = 0.0

    rows = []
    rows.append({
        'Model': model_name,
        'Stage': 'Prefill',
        'Batch': batch_size,
        'Input_Tokens': input_tokens,
        'Output_Tokens': 0,
        **{c: float(prefill_summary[c].values[0]) for c in cols},
    })
    rows.append({
        'Model': model_name,
        'Stage': 'Decode',
        'Batch': batch_size,
        'Input_Tokens': input_tokens,
        'Output_Tokens': output_tokens,
        **{c: float(decode_summary[c].values[0]) for c in cols},
    })

    out_df = pd.DataFrame(rows)

    # Save CSV next to this script
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'mem_metrics_llama2_7b_2x3090_tp2.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved memory metrics to: {out_path}")


if __name__ == '__main__':
    main()






