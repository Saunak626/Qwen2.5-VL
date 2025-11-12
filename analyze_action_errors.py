#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误样本文本动作统计分析脚本
分析错误样本中的行为特征，包括吸吮行为、手脚活动、皱眉等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(action_csv_path, video_csv_path):
    """
    加载行为分析数据和视频错误统计数据
    """
    action_df = pd.read_csv(action_csv_path)
    video_df = pd.read_csv(video_csv_path)
    
    print(f"行为分析数据: {len(action_df)} 条记录")
    print(f"视频错误统计数据: {len(video_df)} 条记录")
    
    return action_df, video_df

def analyze_prediction_errors(action_df):
    """
    分析预测错误的类型分布
    """
    print("\n=== 预测错误类型分析 ===")
    
    # 统计各种错误类型
    error_types = defaultdict(int)
    
    for _, row in action_df.iterrows():
        pred = row['预测标签']
        true = row['真实标签']
        error_type = f"{pred} -> {true}"
        error_types[error_type] += 1
    
    # 按错误数量排序
    sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    
    print("错误类型分布:")
    for error_type, count in sorted_errors:
        print(f"  {error_type}: {count} 次")
    
    return error_types

def analyze_behavior_patterns(action_df):
    """
    分析行为模式特征
    """
    print("\n=== 行为模式分析 ===")
    
    # 基本统计信息
    behavior_cols = ['吸吮行为次数', '吸吮行为时长(秒)', '手脚活动次数', '手脚活动时长(秒)', 
                    '皱眉次数', '皱眉时长(秒)', '总行为次数', '总持续时间(秒)']
    
    print("\n行为特征统计:")
    for col in behavior_cols:
        if col in action_df.columns:
            stats = action_df[col].describe()
            print(f"\n{col}:")
            print(f"  平均值: {stats['mean']:.2f}")
            print(f"  中位数: {stats['50%']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  最大值: {stats['max']:.2f}")
            print(f"  非零样本数: {(action_df[col] > 0).sum()}")
    
    return behavior_cols

def analyze_by_prediction_type(action_df):
    """
    按预测类型分析行为特征
    """
    print("\n=== 按预测错误类型分析行为特征 ===")
    
    # 创建错误类型列
    action_df['错误类型'] = action_df['预测标签'] + ' -> ' + action_df['真实标签']
    
    # 统计主要错误类型的行为特征
    error_type_counts = action_df['错误类型'].value_counts()
    main_error_types = error_type_counts.head(5).index
    
    behavior_cols = ['吸吮行为次数', '手脚活动次数', '皱眉次数', '总行为次数']
    
    for error_type in main_error_types:
        subset = action_df[action_df['错误类型'] == error_type]
        print(f"\n{error_type} ({len(subset)} 个样本):")
        
        for col in behavior_cols:
            if col in subset.columns:
                mean_val = subset[col].mean()
                nonzero_count = (subset[col] > 0).sum()
                nonzero_pct = nonzero_count / len(subset) * 100
                print(f"  {col}: 平均 {mean_val:.2f}, 非零样本 {nonzero_count}/{len(subset)} ({nonzero_pct:.1f}%)")

def analyze_behavior_correlation(action_df):
    """
    分析行为特征之间的相关性
    """
    print("\n=== 行为特征相关性分析 ===")
    
    behavior_cols = ['吸吮行为次数', '吸吮行为时长(秒)', '手脚活动次数', '手脚活动时长(秒)', 
                    '皱眉次数', '皱眉时长(秒)', '总行为次数', '总持续时间(秒)']
    
    # 计算相关性矩阵
    available_cols = [col for col in behavior_cols if col in action_df.columns]
    corr_matrix = action_df[available_cols].corr()
    
    print("\n行为特征相关性矩阵:")
    print(corr_matrix.round(3))
    
    # 找出高相关性的特征对
    print("\n高相关性特征对 (|r| > 0.7):")
    for i in range(len(available_cols)):
        for j in range(i+1, len(available_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                print(f"  {available_cols[i]} <-> {available_cols[j]}: {corr_val:.3f}")

def analyze_video_specific_patterns(action_df):
    """
    分析不同视频的行为模式
    """
    print("\n=== 不同视频的行为模式分析 ===")
    
    video_stats = action_df.groupby('源视频文件名').agg({
        '样本索引': 'count',
        '吸吮行为次数': ['mean', 'std'],
        '手脚活动次数': ['mean', 'std'],
        '皱眉次数': ['mean', 'std'],
        '总行为次数': ['mean', 'std']
    }).round(2)
    
    video_stats.columns = ['错误样本数', '吸吮_均值', '吸吮_标准差', '手脚_均值', '手脚_标准差', 
                          '皱眉_均值', '皱眉_标准差', '总行为_均值', '总行为_标准差']
    
    print("\n各视频错误样本行为特征:")
    print(video_stats)
    
    return video_stats

def generate_behavior_summary(action_df):
    """
    生成行为特征总结报告
    """
    print("\n=== 行为特征总结报告 ===")
    
    total_samples = len(action_df)
    
    # 统计有行为活动的样本
    has_sucking = (action_df['吸吮行为次数'] > 0).sum()
    has_movement = (action_df['手脚活动次数'] > 0).sum()
    has_frowning = (action_df['皱眉次数'] > 0).sum()
    has_any_behavior = (action_df['总行为次数'] > 0).sum()
    
    print(f"总错误样本数: {total_samples}")
    print(f"有吸吮行为的样本: {has_sucking} ({has_sucking/total_samples*100:.1f}%)")
    print(f"有手脚活动的样本: {has_movement} ({has_movement/total_samples*100:.1f}%)")
    print(f"有皱眉行为的样本: {has_frowning} ({has_frowning/total_samples*100:.1f}%)")
    print(f"有任何行为的样本: {has_any_behavior} ({has_any_behavior/total_samples*100:.1f}%)")
    print(f"无任何行为的样本: {total_samples-has_any_behavior} ({(total_samples-has_any_behavior)/total_samples*100:.1f}%)")
    
    # 行为强度分析
    print("\n行为强度分析:")
    high_sucking = (action_df['吸吮行为次数'] >= 5).sum()
    high_movement = (action_df['手脚活动次数'] >= 5).sum()
    high_frowning = (action_df['皱眉次数'] >= 3).sum()
    
    print(f"高吸吮活动样本 (≥5次): {high_sucking} ({high_sucking/total_samples*100:.1f}%)")
    print(f"高手脚活动样本 (≥5次): {high_movement} ({high_movement/total_samples*100:.1f}%)")
    print(f"高皱眉活动样本 (≥3次): {high_frowning} ({high_frowning/total_samples*100:.1f}%)")

def save_analysis_results(action_df, output_path):
    """
    保存分析结果到文件
    """
    # 创建分析结果数据框
    analysis_results = []
    
    # 按错误类型统计
    action_df['错误类型'] = action_df['预测标签'] + ' -> ' + action_df['真实标签']
    error_type_stats = action_df.groupby('错误类型').agg({
        '样本索引': 'count',
        '吸吮行为次数': ['mean', 'std'],
        '手脚活动次数': ['mean', 'std'],
        '皱眉次数': ['mean', 'std'],
        '总行为次数': ['mean', 'std'],
        '总持续时间(秒)': ['mean', 'std']
    }).round(2)
    
    error_type_stats.columns = ['样本数', '吸吮次数_均值', '吸吮次数_标准差', 
                               '手脚活动_均值', '手脚活动_标准差',
                               '皱眉次数_均值', '皱眉次数_标准差',
                               '总行为_均值', '总行为_标准差',
                               '总时长_均值', '总时长_标准差']
    
    error_type_stats.to_csv(output_path.replace('.csv', '_error_type_stats.csv'))
    print(f"\n错误类型统计结果已保存到: {output_path.replace('.csv', '_error_type_stats.csv')}")
    
    # 按视频统计
    video_stats = action_df.groupby('源视频文件名').agg({
        '样本索引': 'count',
        '吸吮行为次数': ['mean', 'std'],
        '手脚活动次数': ['mean', 'std'],
        '皱眉次数': ['mean', 'std'],
        '总行为次数': ['mean', 'std']
    }).round(2)
    
    video_stats.columns = ['错误样本数', '吸吮_均值', '吸吮_标准差', '手脚_均值', '手脚_标准差', 
                          '皱眉_均值', '皱眉_标准差', '总行为_均值', '总行为_标准差']
    
    video_stats.to_csv(output_path.replace('.csv', '_video_stats.csv'))
    print(f"视频统计结果已保存到: {output_path.replace('.csv', '_video_stats.csv')}")

def main():
    parser = argparse.ArgumentParser(description='分析错误样本中的文本动作统计情况')
    parser.add_argument('--action_csv', default='action_error_analysis.csv',
                       help='行为分析CSV文件路径')
    parser.add_argument('--video_csv', default='video_error_stats_with_action.csv',
                       help='视频错误统计CSV文件路径')
    parser.add_argument('--output', default='action_analysis_results.csv',
                       help='分析结果输出文件路径')
    
    args = parser.parse_args()
    
    print("错误样本文本动作统计分析")
    print("=" * 50)
    
    # 加载数据
    action_df, video_df = load_data(args.action_csv, args.video_csv)
    
    # 执行各项分析
    analyze_prediction_errors(action_df)
    analyze_behavior_patterns(action_df)
    analyze_by_prediction_type(action_df)
    analyze_behavior_correlation(action_df)
    analyze_video_specific_patterns(action_df)
    generate_behavior_summary(action_df)
    
    # 保存分析结果
    save_analysis_results(action_df, args.output)
    
    print("\n分析完成!")

if __name__ == '__main__':
    main()