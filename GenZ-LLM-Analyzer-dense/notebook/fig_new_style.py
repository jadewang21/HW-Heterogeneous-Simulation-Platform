#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# === Removed seaborn style and custom font settings to match the target style ===
# plt.style.use('seaborn-v0_8-darkgrid')
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

def calc_relative_error(sim_val, real_val):
    """计算相对误差"""
    if real_val == 0 or np.isnan(sim_val) or np.isnan(real_val):
        return np.nan
    return abs((sim_val - real_val) / real_val) * 100

def validate_data(df_real, df_sim, bench, physical_cols, simulator_cols):
    """验证数据完整性"""
    print(f"\n=== {bench.upper()} 数据验证 ===")
    print(f"物理机数据形状: {df_real.shape}")
    print(f"模拟器数据形状: {df_sim.shape}")
    
    print(f"物理机数据列: {list(df_real.columns)}")
    print(f"模拟器数据列: {list(df_sim.columns)}")
    
    # 检查关键列是否存在
    physical_missing = []
    for metric, col in physical_cols.items():
        if col not in df_real.columns:
            physical_missing.append(col)
    
    simulator_missing = []
    for metric, col in simulator_cols.items():
        if col not in df_sim.columns:
            simulator_missing.append(col)
    
    if physical_missing:
        print(f"❌ 物理机数据缺少列: {physical_missing}")
        return False
    if simulator_missing:
        print(f"❌ 模拟器数据缺少列: {simulator_missing}")
        return False
    
    print(f"✅ 数据列检查通过")
    
    # 显示前几行数据
    print(f"\n物理机数据预览:")
    print(df_real[['conversation_id', 'E2E', 'TTFT', 'TPOT']].head(3))
    print(f"\n模拟器数据预览:")
    print(df_sim[['E2E(ms)', 'TTFT(ms)', 'TPOT(ms)']].head(3))
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for LLM simulation results")
    parser.add_argument("--summary_dir", type=str, 
                        default="../../benchmark-perf/3090-tp2-llama2-7b",
                        help="Directory containing benchmark CSV files")
    
    args = parser.parse_args()
    summary_dir = args.summary_dir
    
    output_dir = os.path.join(summary_dir, "llama2-32-fig")
    os.makedirs(output_dir, exist_ok=True)

    benchmarks = ["chat", "humaneval", "mmlu"]

    # 物理机和模拟器的列名映射
    physical_cols = {
        "TTFT": "TTFT",      # 物理机数据列名
        "TPOT": "TPOT",
        "E2E": "E2E"
    }

    simulator_cols = {
        "TTFT": "TTFT(ms)",   # 模拟器数据列名
        "TPOT": "TPOT(ms)",
        "E2E": "E2E(ms)"
    }

    # 处理每个基准测试
    for bench in benchmarks:
        csv_real = os.path.join(summary_dir, f"{bench}.csv")
        csv_sim = os.path.join(summary_dir, f"{bench}-sim.csv")
    
        if not os.path.exists(csv_real) or not os.path.exists(csv_sim):
            print(f"❌ {bench} 缺少文件:")
            print(f"  物理机: {csv_real} {'存在' if os.path.exists(csv_real) else '不存在'}")
            print(f"  模拟器: {csv_sim} {'存在' if os.path.exists(csv_sim) else '不存在'}")
            continue
        
        # 读取数据
        try:
            df_real = pd.read_csv(csv_real)
            df_sim = pd.read_csv(csv_sim)
        except Exception as e:
            print(f"❌ {bench} 读取数据失败: {e}")
            continue
        
        # 验证数据
        if not validate_data(df_real, df_sim, bench, physical_cols, simulator_cols):
            continue
        
        # 确保两个数据集有相同的行数
        min_rows = min(len(df_real), len(df_sim))
        if len(df_real) != len(df_sim):
            print(f"⚠️ {bench} 数据行数不匹配，使用前{min_rows}行")
        
        df_real = df_real.head(min_rows)
        df_sim = df_sim.head(min_rows)
        
        # 为每个指标生成对比图
        for metric in ["TTFT", "TPOT", "E2E"]:
            errors = []
            real_values = []
            sim_values = []
            
            # 计算每个请求的相对误差
            for rid in range(len(df_real)):
                real_val = df_real[physical_cols[metric]].iloc[rid]
                sim_val = df_sim[simulator_cols[metric]].iloc[rid]
                
                real_values.append(real_val)
                sim_values.append(sim_val)
                
                err = calc_relative_error(sim_val, real_val)
                errors.append(err)
        
            # 过滤掉NaN值进行统计
            valid_errors = [e for e in errors if not np.isnan(e)]
            valid_real = [r for r, e in zip(real_values, errors) if not np.isnan(e)]
            valid_sim = [s for s, e in zip(sim_values, errors) if not np.isnan(e)]
            
            if not valid_errors:
                print(f"❌ {bench} - {metric} 没有有效的误差数据")
                continue
        
            # === 绘图风格改造：与参考风格一致（不指定颜色，使用默认色板；虚线网格；紧凑布局）===
            plt.figure(figsize=(14, 8))
            
            # 子图1: 相对误差
            plt.subplot(2, 1, 1)
            x = range(len(errors))
            line_err, = plt.plot(
                x, errors,
                marker='o', markersize=4, linewidth=1.2,
                linestyle='-', label='Relative Error (%)'
            )
            
            # 20%阈值线（不指定颜色，保持默认；虚线）
            plt.axhline(y=20, linestyle='--', alpha=0.8, label='20% Error Threshold')
            # 填充"可接受区间"使用与误差线相同颜色，透明度降低
            fill_color = line_err.get_color()
            plt.fill_between(x, 0, 20, color=fill_color, alpha=0.12, label='Acceptable Range (≤20%)')
            
            plt.xlabel("Request ID")
            plt.ylabel("Relative Error (%)")
            plt.title(f"{bench.upper()} - {metric} Relative Error (Sim vs Physical)", fontsize=14)
            
            # 统计信息
            mean_err = np.mean(valid_errors)
            median_err = np.median(valid_errors)
            max_err = np.max(valid_errors)
            over_20_pct = sum(1 for e in valid_errors if e > 20) / len(valid_errors) * 100
            
            plt.text(
                0.98, 0.95,
                f"Mean: {mean_err:.1f}%\nMedian: {median_err:.1f}%\nMax: {max_err:.1f}%\n>20%: {over_20_pct:.1f}%",
                transform=plt.gca().transAxes, fontsize=10,
                va='top', ha='right',
                bbox=dict(boxstyle='round', alpha=0.85)  # 默认白底，贴合参考风格
            )
            
            # 设置y轴范围
            max_err_display = min(max_err * 1.1, 200)  # 限制显示范围
            plt.ylim(0, max_err_display)
            plt.legend(title='Metrics')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # 子图2: 绝对值对比
            plt.subplot(2, 1, 2)
            x2 = range(len(real_values))
            plt.plot(
                x2, real_values,
                marker='o', markersize=4, linewidth=1.2,
                linestyle='-', label='Physical Machine'
            )
            plt.plot(
                x2, sim_values,
                marker='x', markersize=4, linewidth=1.2,
                linestyle='--', label='Simulator'
            )
            
            plt.xlabel("Request ID")
            plt.ylabel(f"{metric} (ms)")
            plt.title(f"{bench.upper()} - {metric} Absolute Values Comparison", fontsize=14)
            plt.legend(title='Source')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # 相关性信息
            correlation = np.corrcoef(valid_real, valid_sim)[0, 1] if len(valid_real) > 1 else np.nan
            plt.text(
                0.02, 0.98, f"Correlation: {correlation:.3f}",
                transform=plt.gca().transAxes, fontsize=10,
                va='top', ha='left',
                bbox=dict(boxstyle='round', alpha=0.85)  # 默认白底
            )
            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(output_dir, f"{bench}_{metric}_comparison.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            print(
                f"✅ {bench} - {metric}: Mean Error = {mean_err:.1f}%, "
                f"Correlation = {correlation:.3f}, Saved: {save_path}"
            )

    print(f"\n所有对比图表已保存到: {output_dir}")

if __name__ == "__main__":
    main()
