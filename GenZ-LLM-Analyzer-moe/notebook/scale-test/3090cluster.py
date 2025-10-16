from GenZ import prefill_moddeling, decode_moddeling
import GenZ, Systems.system_configs as cfg
print("GenZ loaded from:", GenZ.__file__)
print("Systems.system_configs from:", cfg.__file__)
print("available systems:", list(cfg.system_configs.keys())[:40], "...")

system_name = "RTX3090_PCIE_G4_IB100G"
model = "llama2_7b"

# 节点内 TP=2，节点间 PP=2
outputs_prefill = prefill_moddeling(
    model=model,
    batch_size=1,
    input_tokens=512,
    system_name=system_name,
    system_eff=0.6,
    bits='bf16',
    tensor_parallel=2,
    pipeline_parallel=2,
    expert_parallel=1,
    collective_strategy='GenZ',
)

outputs_decode = decode_moddeling(
    model=model,
    batch_size=1,
    input_tokens=512,
    output_tokens=512,
    Bb=1,
    system_name=system_name,
    system_eff=0.6,
    bits='bf16',
    tensor_parallel=2,
    pipeline_parallel=2,
    expert_parallel=1,
    collective_strategy='GenZ',
)

# ===== After outputs_prefill / outputs_decode =====
ttft_ms = outputs_prefill["Latency"]
prefill_tps = outputs_prefill["Throughput"]
tpot_ms = outputs_decode["Latency"]
decode_tps = outputs_decode["Throughput"]

print(f"TTFT (ms): {ttft_ms:.3f} | Prefill TPS (tokens/s): {prefill_tps:.2f}")
print(f"TPOT (ms/token): {tpot_ms:.6f} | Decode TPS (tokens/s): {decode_tps:.2f}")

# 估算一个 E2E（输入 512 + 输出 512），E2E ≈ TTFT + TPOT * 输出长度
gen_tokens = 512
e2e_ms = ttft_ms + tpot_ms * gen_tokens
print(f"Estimated E2E for 512 output tokens: {e2e_ms:.2f} ms")

# 如需看通信/算子分解（快速 sanity check）
print("Prefill breakdown:", outputs_prefill["Runtime_breakdown"])
print("Decode  breakdown:", outputs_decode["Runtime_breakdown"])

# 若想更细：比如只看通信占比（Send_Recv/A2A/AR 等）
rb_p = outputs_prefill["Runtime_breakdown"]
rb_d = outputs_decode["Runtime_breakdown"]
print(f"Prefill Comm(ms): {rb_p.Collective:.3f}  (SendRecv={rb_p.Send_Recv_time:.3f}, A2A={rb_p.A2A_time:.3f}, AR={rb_p.AR_time:.3f})")
print(f"Decode  Comm(ms): {rb_d.Collective:.3f}  (SendRecv={rb_d.Send_Recv_time:.3f}, A2A={rb_d.A2A_time:.3f}, AR={rb_d.AR_time:.3f})")

