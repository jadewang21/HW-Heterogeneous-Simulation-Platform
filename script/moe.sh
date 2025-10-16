echo "此脚本执行6节点48卡RTX4090推理DeepSeek-V3-671B模型性能预测"

#设置环境变量
export PYTHONPATH="$(dirname $(pwd))/GenZ-LLM-Analyzer-moe:$PYTHONPATH"

echo "PYTHONPATH set to:"
echo "$PYTHONPATH"

cd ../GenZ-LLM-Analyzer-moe/notebook/scale-test

python deepseek_v3_4n8g.py

echo "性能预测完成，结果可与《G5208 DS-R1-Cluster DeepSeek大模型推性能白皮书》作对比"
