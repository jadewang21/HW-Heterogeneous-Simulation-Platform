#!/usr/bin/env python3
import os
import time
import warnings
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_mem_mb(device: int = 0) -> float:
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def build_prompt(tokenizer, target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    text = base
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += base
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


def main():
    warnings.filterwarnings("ignore")

    model_path = "/home/wang/model/llama2-7b-hf"
    device = 0
    input_tokens = 4096
    output_tokens = 128

    assert torch.cuda.is_available(), "CUDA不可用，请在有GPU的环境运行"

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    sync()
    base_mb = get_mem_mb(device)

    # 加载模型与分词器（FP16，单卡）
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    sync()
    mem_after_load_mb = get_mem_mb(device)
    weights_mb = max(mem_after_load_mb - base_mb, 0.0)

    prompt = build_prompt(tokenizer, input_tokens)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # 显式 Prefill，保留 past_key_values 以确保KV常驻显存
    torch.cuda.empty_cache()
    sync()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_kv = outputs.past_key_values  # 保留引用，防止释放

    sync()
    mem_after_prefill_mb = get_mem_mb(device)
    kv_prefill_mb = max(mem_after_prefill_mb - weights_mb, 0.0)

    # 显式 Decode：基于 cache 单步滚动生成 output_tokens 次
    cur_input_ids = input_ids[:, -1:]
    for _ in range(output_tokens):
        with torch.inference_mode():
            out = model(
                input_ids=cur_input_ids,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_kv,
            )
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            past_kv = out.past_key_values  # 更新并持有引用
            cur_input_ids = next_token

    sync()
    mem_after_decode_mb = get_mem_mb(device)
    kv_total_mb = max(mem_after_decode_mb - weights_mb, 0.0)
    kv_decode_delta_mb = max(mem_after_decode_mb - mem_after_prefill_mb, 0.0)

    # 保存结果
    out_dir = Path(__file__).parent
    out_path = out_dir / "single_gpu_kv_metrics_llama2_7b.csv"
    df = pd.DataFrame([
        {
            "Model": "llama2-7b-hf",
            "GPU_ID": device,
            "Input_Tokens": input_tokens,
            "Output_Tokens": output_tokens,
            "Weights (MB)": round(weights_mb, 2),
            "KV Prefill (MB)": round(kv_prefill_mb, 2),
            "KV Total (MB)": round(kv_total_mb, 2),
            "KV Decode Delta (MB)": round(kv_decode_delta_mb, 2),
        }
    ])
    df.to_csv(out_path, index=False)
    print(f"已保存单卡显存测量CSV: {out_path}")


if __name__ == "__main__":
    main()


