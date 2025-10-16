echo "此脚本执行双卡RTX 3090推理Llama2-7b模型的性能预测，并与真实值对比生成图表"

#设置环境变量
export PYTHONPATH="$(dirname $(pwd))/GenZ-LLM-Analyzer-dense:$PYTHONPATH"

echo "PYTHONPATH set to:"
echo "$PYTHONPATH"

cd ../GenZ-LLM-Analyzer-dense/notebook

python llm_simulation.py

echo "性能预测完成，结果保存在HW-Heterogeneous-Simulation-Platform/benchmark-perf/3090-tp2-llama2-7b/chat-sim.csv"

cd ../GenZ-LLM-Analyzer-dense/notebook

python fig_new_style.py

echo "图表生成完成，结果保存在HW-Heterogeneous-Simulation-Platform/benchmark-perf/3090-tp2-llama2-7b/llama2-32-fig"