#!/bin/bash
# 设置正确的Python路径并运行脚本

# 设置PYTHONPATH，确保优先使用当前目录的GenZ模块
export PYTHONPATH="/home/wang/sim/delivery/GenZ-LLM-Analyzer-dense:$PYTHONPATH"

# 进入notebook目录
cd /home/wang/sim/delivery/GenZ-LLM-Analyzer-dense/notebook

# 运行脚本，传递所有参数
python3 3090_llama2_32.py "$@"
