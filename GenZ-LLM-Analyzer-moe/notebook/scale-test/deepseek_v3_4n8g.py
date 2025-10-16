import os, sys
# 确保优先使用当前仓库的 GenZ 实现
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURR_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from GenZ import prefill_moddeling, decode_moddeling
import GenZ, Systems.system_configs as cfg

print("GenZ loaded from:", GenZ.__file__)
print("Systems.system_configs from:", cfg.__file__)

system_name = "RTX4090_PCIE_G4_IB25G"
model = "deepseek-ai/DeepSeek-V3-Base"

# 集群：4 节点 × 8 卡。目标：pp=4, tp=8（并使 EP=8 与 TP 共享分组）
# 注意：通过 ep_share_tp_group=True，使 MoE 专家并行与 TP 同组；
# Gate 不做 TP 切分（在 ffn.py 中已实现）；FFN 仍按 TP 切分并做 AR。

common_kwargs = dict(
    model=model,
    batch_size=1,
    input_tokens=80,
    system_name=system_name,
    system_eff=0.3,
    bits='fp8',
    tensor_parallel=8,
    pipeline_parallel=6,
    expert_parallel=8,
    ep_share_tp_group=True,
    collective_strategy='GenZ',
)

outputs_prefill = prefill_moddeling(**common_kwargs)

outputs_decode = decode_moddeling(
    **common_kwargs,
    output_tokens=512,
    Bb=1,
)

ttft_ms = outputs_prefill["Latency"]
prefill_tps = outputs_prefill["Throughput"]
tpot_ms = outputs_decode["Latency"]
decode_tps = outputs_decode["Throughput"]

print(f"TTFT (ms): {ttft_ms:.3f} | Prefill TPS (tokens/s): {prefill_tps:.2f}")
print(f"TPOT (ms/token): {tpot_ms:.6f} | Decode TPS (tokens/s): {decode_tps:.2f}")

gen_tokens = 512
e2e_ms = ttft_ms + tpot_ms * gen_tokens
print(f"Estimated E2E for {gen_tokens} output tokens: {e2e_ms:.2f} ms")

rb_p = outputs_prefill["Runtime_breakdown"]
rb_d = outputs_decode["Runtime_breakdown"]
print(f"Prefill Comm(ms): {rb_p.Collective:.3f}  (SendRecv={rb_p.Send_Recv_time:.3f}, A2A={rb_p.A2A_time:.3f}, AR={rb_p.AR_time:.3f})")
print(f"Decode  Comm(ms): {rb_d.Collective:.3f}  (SendRecv={rb_d.Send_Recv_time:.3f}, A2A={rb_d.A2A_time:.3f}, AR={rb_d.AR_time:.3f})")


