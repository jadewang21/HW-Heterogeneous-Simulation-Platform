# GenZ-LLM-Analyzer 大语言模型推理性能分析平台

一个用于大语言模型（LLM）推理性能建模和分析的综合框架，支持Dense模型和MoE（专家混合）架构的性能预测。

## 📋 项目概述

GenZ-LLM-Analyzer提供准确的LLM推理性能预测，支持各种硬件配置，包括单GPU、多GPU集群和专用加速器。框架支持Dense模型（如Llama系列）和MoE模型（如DeepSeek系列）的性能分析。

## 📁 项目结构

```
delivery/
├── benchmark-mem/          # 内存分析测试项的物理机真实数据
├── benchmark-perf/         # 性能预测测试项的物理机真实数据  
├── GenZ-LLM-Analyzer-dense/ # Dense模型预测仿真平台（Llama系列等）
├── GenZ-LLM-Analyzer-moe/  # MoE模型预测仿真平台（DeepSeek系列等）
└── README.md
```

## 🚀 快速开始

### 依赖安装

首先安装项目所需的依赖包：

```bash
# 安装Python依赖
pip install -r requirements.txt
```

### 环境配置

```bash
# 设置Python路径
export PYTHONPATH={YOUR_PATH}/delivery/GenZ-LLM-Analyzer-dense:$PYTHONPATH
```

### 内存分析

执行内存使用模式分析：

```bash
python delivery/GenZ-LLM-Analyzer-dense/notebook/mem-test/mem_profile_llama2_7b_2x3090_tp1.py
```

即可在当前目录生成llama2-7b模型样例request的内存占用分析结果，可与`delivery/benchmark-mem/single_gpu_kv_metrics_llama2_7b.csv`真实物理机测试数据作对比。

### 性能预测

#### Dense模型（Llama系列等）

对dense模型（如Llama系列）的性能预测，需要在GenZ-LLM-Analyzer-dense路径中进行，并且在运行下述启动器llm_simulation.py前需要执行：

```bash
export PYTHONPATH={YOUR_PATH}/delivery/GenZ-LLM-Analyzer-dense:$PYTHONPATH
```

性能预测前需要在物理机采集数据，并以chat.csv的形式存放在特定目录下，将存放物理机数据的目录输入到启动脚本即可开始仿真。

**使用示例：**

1. **查看所有可用参数**：
```bash
python delivery/GenZ-LLM-Analyzer-dense/notebook/llm_simulation.py --help
```

**主要参数说明**：

- `--model`: 要仿真的模型名称（默认：llama2_7b）
- `--quantization`: 量化设置，支持bf16/int8/int4等（默认：bf16）
- `--beam_size`: Beam搜索大小，影响解码质量（默认：1）
- `--batch_size`: 仿真批次大小（默认：1）
- `--system_eff`: 系统效率因子，0-1之间（默认：0.6）
- `--tp_nodes`: 张量并行节点数（默认：2）
- `--pp_nodes`: 流水线并行节点数（默认：1）
- `--system`: 系统配置名称，如RTX3090_GPU/H100_GPU等（默认：RTX3090_GPU）
- `--comm_backend`: 通信后端类型，nv_like/pcie_like（默认：pcie_like）
- `--summary_dir`: 包含基准测试CSV文件的目录路径

2. **进行单卡RTX3090推理llama2-7b的仿真**：
```bash
python delivery/GenZ-LLM-Analyzer-dense/notebook/llm_simulation.py \
    --tp_nodes 1 \
    --system RTX3090_GPU \
    --summary_dir ../../benchmark-perf/3090-tp1-llama2-7b
```

3. **获取预测的性能结果，保存在benchmark-perf/3090-tp1-llama2-7b，如需可视化**：
```bash
python delivery/GenZ-LLM-Analyzer-dense/notebook/fig_new_style.py \
    --summary_dir ../../benchmark-perf/3090-tp1-llama2-7b
```

可视化结果保存在`delivery/benchmark-perf/3090-tp1-llama2-7b/llama2-32-fig`

#### MoE模型（DeepSeek系列等）

对MoE模型（如DeepSeek系列）的性能预测，需要在GenZ-LLM-Analyzer-moe路径中进行，并且在运行下述启动器deepseek_v3_4n8g.py前需要执行：

```bash
export PYTHONPATH={YOUR_PATH}/delivery/GenZ-LLM-Analyzer-moe:$PYTHONPATH
```

MoE模型目前仅支持DeepSeek-V3-671B，支持任意规模的RTX3090与RTX4090集群的仿真，执行：

```bash
python delivery/GenZ-LLM-Analyzer-moe/notebook/scale-test/deepseek_v3_4n8g.py
```

即可在终端输出6节点，单节点8卡，pp = 6，tp = 8的4090集群推理非量化DeepSeek-V3-671B模型的性能预测数据，结果可与《G5208DS-R1-Cluster DeepSeek大模型推性能白皮书》中结果对比，如需更改配置，可直接修改deepseek_v3_4n8g.py中的common_kwargs字典。

## 🔧 系统配置

### 支持的硬件配置

| 系统配置 | FLOPS (T) | 内存 (GB) | 内存带宽 (GB/s) | 互连带宽 (GB/s) |
|---------|-----------|-----------|-----------------|-----------------|
| A100_40GB_GPU | 312 | 40 | 1600 | 150 |
| A100_80GB_GPU | 312 | 80 | 2039 | 150 |
| H100_GPU | 989 | 80 | 3400 | 450 |
| RTX3090_GPU | 142 | 24 | 936 | 16 |
| RTX4090_GPU | 330 | 24 | 1008 | 32 |
| MI300X | 1307 | 192 | 5300 | 400 |
| TPUv6 | 926 | 32 | 1640 | 100 |

### 自定义系统配置

如需添加修改系统配置，可以在`delivery/GenZ-LLM-Analyzer-dense/Systems/system_configs.py`按照已有格式任意添加：

```python
system_configs: Dict[str, Dict[str, Any]] = {
    'CUSTOM_GPU': {
        'Flops': 500,           # TFLOPs
        'Memory_size': 32,      # GB
        'Memory_BW': 1000,      # GB/s
        'ICN': 100,            # GB/s
        'L2_Cache_Size': 50,   # MB
        'real_values': True
    }
}
```

## 📊 仿真参数详细说明

### 模型配置参数

- `--model`: 要仿真的模型名称
  - 支持：llama2_7b, llama2_13b, llama2_70b, gpt3_175b等
  - 默认：llama2_7b

- `--quantization`: 量化设置
  - 支持：bf16, int8, int4, fp16等
  - 默认：bf16
  - 影响：计算精度和内存使用

- `--beam_size`: Beam搜索大小
  - 范围：1-8
  - 默认：1
  - 影响：解码质量和计算复杂度

- `--batch_size`: 仿真批次大小
  - 范围：1-32
  - 默认：1
  - 影响：内存使用和并行效率

### 并行化配置参数

- `--tp_nodes`: 张量并行节点数
  - 范围：1-8
  - 默认：2
  - 作用：将模型权重分割到多个GPU

- `--pp_nodes`: 流水线并行节点数
  - 范围：1-8
  - 默认：1
  - 作用：将模型层分割到多个GPU

### 系统配置参数

- `--system`: 系统配置名称
  - 支持：RTX3090_GPU, RTX4090_GPU, A100_40GB_GPU, H100_GPU等
  - 默认：RTX3090_GPU
  - 作用：指定硬件配置

- `--system_eff`: 系统效率因子
  - 范围：0.1-1.0
  - 默认：0.6
  - 作用：模拟实际系统的效率损失

- `--comm_backend`: 通信后端类型
  - 选项：nv_like（NVIDIA风格）, pcie_like（PCIe风格）
  - 默认：pcie_like
  - 作用：影响多GPU通信性能

### 数据配置参数

- `--summary_dir`: 包含基准测试CSV文件的目录路径
  - 格式：相对或绝对路径
  - 要求：目录下需包含chat.csv文件
  - CSV文件需包含input_tokens和output_tokens列

## 📈 输出结果分析

### 性能指标

- **TTFT (Time To First Token)**: 处理输入序列的首token延迟
- **TPOT (Time Per Output Token)**: 生成每个输出token的时间
- **E2E (End-to-End)**: 端到端总推理延迟
- **吞吐量**: Prefill和Decode阶段的每秒token数

### 可视化结果

框架生成全面的对比图表：

- 仿真结果与物理机结果的相对误差分析
- 绝对值对比
- 统计相关性分析
- 按请求的性能分解

结果保存在`{summary_dir}/llama2-32-fig/`目录中，包含详细的PNG可视化图表。

## 🧪 测试验证

### 内存分析测试

```bash
python delivery/GenZ-LLM-Analyzer-dense/notebook/mem-test/mem_profile_llama2_7b_2x3090_tp1.py
```

### 性能仿真测试

```bash
# 测试不同配置
python delivery/GenZ-LLM-Analyzer-dense/notebook/llm_simulation.py \
    --model llama2_7b \
    --system RTX3090_GPU \
    --tp_nodes 2 \
    --summary_dir ../../benchmark-perf/3090-tp2-llama2-7b
```

## 📚 高级用法

### 多节点集群仿真

对于大规模集群仿真：

```bash
# DeepSeek-V3-671B在6节点集群上
python delivery/GenZ-LLM-Analyzer-moe/notebook/scale-test/deepseek_v3_4n8g.py
```

### 自定义模型集成

1. 在`GenZ/Models/`中定义模型配置
2. 添加硬件特定优化
3. 配置并行化策略
4. 使用自定义参数运行仿真
