# DeepSeek-R1 模型评估指南

基于GenZ-LLM-Analyzer框架的完整评估教程

## 目录
1. [项目概述](#项目概述)
2. [核心概念](#核心概念)
3. [环境准备](#环境准备)
4. [评估新模型的完整流程](#评估新模型的完整流程)
5. [DeepSeek-R1评估实例](#deepseek-r1评估实例)
6. [性能指标解释](#性能指标解释)
7. [故障排除](#故障排除)
8. [高级用法](#高级用法)

## 项目概述

GenZ-LLM-Analyzer是一个用于分析大语言模型在不同硬件平台上推理性能的框架。它可以帮助你：
- 预测模型在特定硬件上的延迟和吞吐量
- 对比不同模型、硬件、量化方式的性能
- 优化部署策略和资源配置
- 分析内存使用和计算瓶颈

## 核心概念

### 关键性能指标

- **TTFT (Time to First Token)**: 首个token生成时间，反映预填充阶段性能
- **TPOT (Time per Output Token)**: 每个输出token的平均生成时间
- **吞吐量 (Throughput)**: 每秒处理的token数量
- **延迟 (Latency)**: 完成整个推理过程的时间

### 评估阶段

1. **预填充 (Prefill)**: 处理输入序列，生成KV缓存
2. **解码 (Decode)**: 逐个生成输出token

### 硬件抽象

GenZ支持多种硬件平台：
- **NVIDIA GPU**: A100, H100, GH200等
- **AMD GPU**: MI300X, MI325X等
- **Intel加速器**: Gaudi3等
- **Google TPU**: TPUv4, TPUv5e, TPUv6等

## 环境准备

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/abhibambhaniya/GenZ-LLM-Analyzer.git
cd GenZ-LLM-Analyzer

# 安装依赖
pip install -r requirements.txt
# 或使用conda
conda env create -f environment.yml
conda activate genz
```

### 验证安装

```python
from GenZ import decode_moddeling, prefill_moddeling, get_configs
from Systems.system_configs import system_configs
print("GenZ安装成功!")
```

## 评估新模型的完整流程

### 步骤1: 定义模型架构

创建模型配置，需要指定以下关键参数：

```python
from GenZ import ModelConfig

deepseek_r1_7b = ModelConfig(
    model='deepseek-r1-7b',
    vocab_size=102400,           # 词汇表大小
    max_model_len=128000,        # 最大序列长度
    hidden_size=4096,            # 隐藏层维度
    intermediate_size=11008,     # FFN中间层大小
    num_decoder_layers=32,       # 解码器层数
    num_attention_heads=32,      # 注意力头数
    num_key_value_heads=32,      # KV头数（MHA/GQA/MQA）
    head_dim=128,                # 每个注意力头的维度
    hidden_act="silu"            # 激活函数
)
```

### 步骤2: 配置评估参数

```python
# 硬件配置
hardware = 'H100_GPU'  # 或其他支持的硬件

# 量化设置
quantization = 'int8'  # fp32, fp16, bf16, int8, int4, int2

# 工作负载参数
input_tokens = 2048    # 输入序列长度
output_tokens = 512    # 输出序列长度
batch_size = 4         # 批处理大小
beam_size = 1          # 束搜索大小
```

### 步骤3: 运行评估

```python
from GenZ import prefill_moddeling, decode_moddeling
from Systems.system_configs import system_configs

# 获取系统配置
system_config = system_configs[hardware]

# 预填充阶段评估
prefill_result = prefill_moddeling(
    model='deepseek-r1-7b',
    batch_size=batch_size,
    input_tokens=input_tokens,
    system_name=system_config,
    system_eff=0.8,  # 系统效率
    bits=quantization,
    tensor_parallel=1,  # 张量并行度
    debug=False
)

# 解码阶段评估
decode_result = decode_moddeling(
    model='deepseek-r1-7b',
    batch_size=batch_size,
    Bb=beam_size,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    system_name=system_config,
    system_eff=0.8,
    bits=quantization,
    tensor_parallel=1,
    debug=False
)
```

### 步骤4: 分析结果

```python
# 提取关键指标
prefill_latency = prefill_result['Latency']  # ms
decode_latency = decode_result['Latency']    # ms
decode_throughput = decode_result['Throughput']  # tokens/s

# 计算端到端指标
total_latency = prefill_latency + decode_latency
ttft = prefill_latency
tpot = decode_latency / output_tokens
end_to_end_throughput = (output_tokens * batch_size) / (total_latency / 1000)

print(f"TTFT: {ttft:.2f} ms")
print(f"TPOT: {tpot:.2f} ms")
print(f"端到端吞吐量: {end_to_end_throughput:.2f} tokens/s")
```

## DeepSeek-R1评估实例

我已经为你创建了完整的DeepSeek-R1评估脚本 `deepseek_r1_evaluation.py`，包含以下功能：

### 主要特性

1. **完整的模型定义**: 支持7B、32B、671B三个规模
2. **多硬件平台支持**: H100、A100、GH200、MI300X等
3. **多种量化选项**: FP16、INT8、INT4等
4. **多样化测试场景**: 问答、聊天、代码生成、长文本处理
5. **详细性能分析**: 包括可视化图表和部署建议

### 使用方法

```bash
cd /home/wang/sim/org-genz/GenZ-LLM-Analyzer/notebook
python deepseek_r1_evaluation.py
```

### 脚本结构

```python
class DeepSeekR1Evaluator:
    def __init__(self):
        # 初始化模型配置和测试场景
        
    def register_deepseek_models(self):
        # 将模型注册到GenZ系统
        
    def evaluate_single_config(self, ...):
        # 评估单个配置的性能
        
    def run_comprehensive_evaluation(self, model_name):
        # 运行全面的性能评估
        
    def analyze_results(self, results_df):
        # 分析评估结果
        
    def create_performance_plots(self, results_df):
        # 创建性能可视化图表
        
    def generate_recommendation_report(self, results_df):
        # 生成部署建议报告
```

## 性能指标解释

### 延迟相关指标

- **预填充延迟**: 处理输入prompt所需时间，通常与输入长度成正比
- **解码延迟**: 生成所有输出token的总时间
- **TTFT**: 用户体验的关键指标，影响感知响应速度
- **TPOT**: 稳态生成速度，影响长输出的总时间

### 吞吐量指标

- **预填充吞吐量**: 预填充阶段处理token的速度
- **解码吞吐量**: 解码阶段生成token的速度  
- **端到端吞吐量**: 整个推理过程的有效token生成速度

### 资源利用率

- **内存使用**: 模型权重 + KV缓存 + 激活值
- **计算利用率**: 实际FLOPS vs 峰值FLOPS的比值
- **带宽利用率**: 内存访问速度vs硬件峰值带宽

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减小批处理大小
   - 使用更积极的量化（INT4/INT8）
   - 增加张量并行度分片模型

2. **模型配置错误**
   - 检查模型参数是否正确
   - 确认vocab_size、hidden_size等核心参数
   - 验证层数和注意力头配置

3. **性能异常**
   - 检查系统效率设置（通常0.7-0.9）
   - 确认硬件配置参数正确
   - 验证并行化策略设置

### 调试技巧

```python
# 启用详细调试信息
result = prefill_moddeling(
    # ... 其他参数
    debug=True  # 启用调试模式
)

# 检查中间结果
from GenZ import get_model_df, get_summary_table
model_df = get_model_df(model_name)
summary = get_summary_table(model_name)
```

## 高级用法

### 自定义硬件配置

```python
custom_hardware = {
    'Flops': 2000,      # TFLOPS
    'Memory_size': 96,  # GB  
    'Memory_BW': 4000,  # GB/s
    'ICN': 600,         # GB/s interconnect
    'real_values': True
}

# 使用自定义硬件评估
result = prefill_moddeling(
    model='deepseek-r1-7b',
    system_name=custom_hardware,
    # ... 其他参数
)
```

### 批量对比分析

```python
models = ['deepseek-r1-7b', 'llama2_7b', 'mistral_7b']
hardware_list = ['H100_GPU', 'A100_80GB_GPU']

results = []
for model in models:
    for hw in hardware_list:
        # 运行评估并收集结果
        result = evaluate_single_config(model, hw, ...)
        results.append(result)

# 创建对比图表
import plotly.express as px
df = pd.DataFrame(results)
fig = px.bar(df, x='model', y='throughput', color='hardware')
fig.show()
```

### 参数扫描优化

```python
# 寻找最佳批处理大小
batch_sizes = [1, 2, 4, 8, 16, 32]
best_config = None
best_throughput = 0

for batch_size in batch_sizes:
    result = evaluate_single_config(
        model_name='deepseek-r1-7b',
        batch_size=batch_size,
        # ... 其他固定参数
    )
    
    if result['e2e_throughput'] > best_throughput:
        best_throughput = result['e2e_throughput']
        best_config = batch_size

print(f"最佳批处理大小: {best_config}")
```

## 总结

通过GenZ-LLM-Analyzer框架，你可以：

1. **快速评估新模型**: 只需定义模型架构即可获得性能预测
2. **对比不同配置**: 系统化地比较硬件、量化、并行化选项
3. **优化部署策略**: 基于数据驱动的决策选择最佳配置
4. **预测资源需求**: 提前了解内存和计算资源需求

使用提供的`deepseek_r1_evaluation.py`脚本作为起点，你可以轻松扩展到评估其他新模型。记住根据实际的模型规格调整配置参数，以获得最准确的性能预测。
