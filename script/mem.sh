echo "此脚本执行单卡RTX 3090推理Llama2-7b模型的内存分析，并将结果保存为表格"

#设置环境变量
export PYTHONPATH="$(dirname $(pwd))/GenZ-LLM-Analyzer-dense:$PYTHONPATH"

echo "PYTHONPATH set to:"
echo "$PYTHONPATH"

cd ../GenZ-LLM-Analyzer-dense/notebook/mem-test

python mem_profile_llama2_7b_2x3090_tp2.py

echo "内存分析完成，分析结果保存在HW-Heterogeneous-Simulation-Platform/benchmark-mem/single_gpu_kv_metrics_llama2_7b_sim.csv"


echo "真实值参考保存在HW-Heterogeneous-Simulation-Platform/benchmark-mem/single_gpu_kv_metrics_llama2_7b.csv"