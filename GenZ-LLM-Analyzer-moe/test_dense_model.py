#!/usr/bin/env python3
"""
测试Dense模型的Flash Attention修正
使用LLaMA模型进行测试
"""

import os, sys
# 确保优先使用当前仓库的 GenZ 实现
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURR_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from GenZ import prefill_moddeling, decode_moddeling
from GenZ.Models.flash_attention_config import set_flash_attention_mode, is_corrected_flash_attention_enabled
import GenZ, Systems.system_configs as cfg

print("=" * 80)
print("测试Dense模型的Flash Attention修正")
print("=" * 80)

system_name = "RTX4090_PCIE_G4_IB25G"
model = "meta-llama/Llama-2-7b-hf"  # 使用LLaMA模型（Dense模型）

# 基础配置
base_kwargs = dict(
    model=model,
    batch_size=1,
    input_tokens=80,
    system_name=system_name,
    system_eff=0.3,
    bits='fp8',
    tensor_parallel=8,
    pipeline_parallel=6,
    expert_parallel=1,  # Dense模型不使用expert_parallel
    ep_share_tp_group=False,
    collective_strategy='GenZ',
    sequence_parallel=2,  # 使用sequence_parallel>1来看到差异
)

print(f"测试配置:")
print(f"  模型: {model} (Dense模型)")
print(f"  输入tokens: {base_kwargs['input_tokens']}")
print(f"  sequence_parallel: {base_kwargs['sequence_parallel']}")
print(f"  expert_parallel: {base_kwargs['expert_parallel']}")

# 测试原始实现
print(f"\n=== 测试原始实现 ===")
set_flash_attention_mode(use_corrected=False)
print(f"使用修正实现: {is_corrected_flash_attention_enabled()}")

try:
    outputs_original = prefill_moddeling(**base_kwargs)
    ttft_original = outputs_original["Latency"]
    print(f"TTFT: {ttft_original:.3f} ms")
except Exception as e:
    print(f"原始实现失败: {e}")
    exit(1)

# 测试修正实现
print(f"\n=== 测试修正实现 ===")
set_flash_attention_mode(use_corrected=True)
print(f"使用修正实现: {is_corrected_flash_attention_enabled()}")

try:
    outputs_corrected = prefill_moddeling(**base_kwargs)
    ttft_corrected = outputs_corrected["Latency"]
    print(f"TTFT: {ttft_corrected:.3f} ms")
    
    # 计算差异
    diff = ttft_corrected - ttft_original
    pct = (diff / ttft_original) * 100 if ttft_original > 0 else 0
    
    print(f"\n差异:")
    print(f"  绝对差异: {diff:+.3f} ms")
    print(f"  相对差异: {pct:+.2f}%")
    
    if abs(diff) < 0.001:
        print(f"  → 结果几乎相同 (差异 < 0.001ms)")
        print(f"  → 这可能意味着修改没有生效")
    elif diff > 0:
        print(f"  → 修正实现更慢 (符合预期)")
        print(f"  → 修改已经生效！")
    else:
        print(f"  → 修正实现更快 (不符合预期)")
        
except Exception as e:
    print(f"修正实现失败: {e}")
    exit(1)

print(f"\n" + "=" * 80)
print("测试完成")
print("=" * 80)
print("说明:")
print("- Dense模型（如LLaMA）应该使用Flash Attention修正")
print("- MoE模型（如DeepSeek-V3）不会使用Flash Attention修正")
print("- 要看到差异，需要使用sequence_parallel>1的配置")

