#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成每个源视频错误次数的CSV统计脚本

功能：
1. 读取测试数据集和预测结果文件
2. 从视频路径中提取源视频文件名
3. 按源视频分组统计错误率
4. 生成CSV格式的统计报告，包含各类标签的详细统计
"""

import json
import csv
import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_state_from_json_string(json_string: str) -> str:
    """从JSON字符串中解析state字段"""
    try:
        data = json.loads(json_string)
        return data.get('state', '').strip()
    except (json.JSONDecodeError, AttributeError):
        return json_string.strip()

def extract_source_video_name(video_path: str) -> str:
    """从视频路径中提取源视频文件名
    
    例如：240717(9.03.26）_clip_300.mp4 -> 240717(9.03.26）
    """
    # 获取文件名（去掉路径）
    filename = os.path.basename(video_path)
    
    # 移除文件扩展名
    filename_without_ext = os.path.splitext(filename)[0]
    
    # 使用正则表达式提取源视频名称（去掉_clip_xxx部分）
    match = re.match(r'^(.+?)_clip_\d+$', filename_without_ext)
    if match:
        return match.group(1)
    
    # 如果没有匹配到_clip_模式，返回原文件名（去掉扩展名）
    return filename_without_ext

def load_test_data(dataset_path: str) -> List[Dict]:
    """加载测试数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_predictions(predictions_path: str) -> List[Dict]:
    """加载预测结果"""
    predictions = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions

def generate_video_error_stats(dataset_path: str, predictions_path: str, output_path: str):
    """生成每个源视频错误次数的CSV统计"""
    
    # 定义标签类别
    label_categories = ['饱腹', '中性', '微饿', '饥饿']
    
    # 加载数据
    print(f"加载测试数据集: {dataset_path}")
    test_data = load_test_data(dataset_path)
    
    print(f"加载预测结果: {predictions_path}")
    predictions = load_predictions(predictions_path)
    
    # 检查数据长度是否匹配
    if len(test_data) != len(predictions):
        print(f"警告: 测试数据集样本数({len(test_data)}) 与预测结果数({len(predictions)}) 不匹配")
        min_len = min(len(test_data), len(predictions))
        test_data = test_data[:min_len]
        predictions = predictions[:min_len]
        print(f"使用前 {min_len} 个样本进行统计")
    
    # 按源视频分组统计
    video_stats = defaultdict(lambda: {
        'total': 0, 
        'errors': 0, 
        'samples': [],
        'label_stats': {label: {'total': 0, 'errors': 0} for label in label_categories}
    })
    
    for i, (sample, pred) in enumerate(zip(test_data, predictions)):
        # 获取视频路径
        videos = sample.get('videos', [])
        if not videos:
            print(f"警告: 样本 {i} 没有视频信息")
            continue
        
        # 提取源视频名称（使用第一个视频）
        video_path = videos[0]
        source_video = extract_source_video_name(video_path)
        
        # 获取真实标签和预测结果
        true_label = sample['messages'][-1]['content'].strip()
        pred_label = parse_state_from_json_string(pred.get('predict', '')).strip()
        
        # 统计总体
        video_stats[source_video]['total'] += 1
        video_stats[source_video]['samples'].append({
            'index': i,
            'video_path': video_path,
            'true_label': true_label,
            'pred_label': pred_label,
            'is_error': true_label != pred_label
        })
        
        if true_label != pred_label:
            video_stats[source_video]['errors'] += 1
        
        # 统计各类标签
        if true_label in label_categories:
            video_stats[source_video]['label_stats'][true_label]['total'] += 1
            if true_label != pred_label:
                video_stats[source_video]['label_stats'][true_label]['errors'] += 1
    
    # 生成CSV报告
    print(f"生成CSV统计报告: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            '源视频文件名', '总样本数', '总错误数', '总错误率(%)',
            '饱腹_总数', '饱腹_错误数', '饱腹_错误率(%)',
            '中性_总数', '中性_错误数', '中性_错误率(%)',
            '微饿_总数', '微饿_错误数', '微饿_错误率(%)',
            '饥饿_总数', '饥饿_错误数', '饥饿_错误率(%)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 按错误率降序排列
        sorted_videos = sorted(video_stats.items(), 
                             key=lambda x: x[1]['errors'] / x[1]['total'] if x[1]['total'] > 0 else 0, 
                             reverse=True)
        
        total_samples = 0
        total_errors = 0
        total_label_stats = {label: {'total': 0, 'errors': 0} for label in label_categories}
        
        for source_video, stats in sorted_videos:
            error_rate = (stats['errors'] / stats['total'] * 100) if stats['total'] > 0 else 0
            
            # 构建行数据
            row_data = {
                '源视频文件名': source_video,
                '总样本数': stats['total'],
                '总错误数': stats['errors'],
                '总错误率(%)': f"{error_rate:.2f}"
            }
            
            # 添加各类标签统计
            for label in label_categories:
                label_total = stats['label_stats'][label]['total']
                label_errors = stats['label_stats'][label]['errors']
                label_error_rate = (label_errors / label_total * 100) if label_total > 0 else 0
                
                row_data[f'{label}_总数'] = label_total
                row_data[f'{label}_错误数'] = label_errors
                row_data[f'{label}_错误率(%)'] = f"{label_error_rate:.2f}"
                
                # 累计总统计
                total_label_stats[label]['total'] += label_total
                total_label_stats[label]['errors'] += label_errors
            
            writer.writerow(row_data)
            
            total_samples += stats['total']
            total_errors += stats['errors']
        
        # 添加总计行
        overall_error_rate = (total_errors / total_samples * 100) if total_samples > 0 else 0
        total_row = {
            '源视频文件名': '总计',
            '总样本数': total_samples,
            '总错误数': total_errors,
            '总错误率(%)': f"{overall_error_rate:.2f}"
        }
        
        # 添加各类标签总计
        for label in label_categories:
            label_total = total_label_stats[label]['total']
            label_errors = total_label_stats[label]['errors']
            label_error_rate = (label_errors / label_total * 100) if label_total > 0 else 0
            
            total_row[f'{label}_总数'] = label_total
            total_row[f'{label}_错误数'] = label_errors
            total_row[f'{label}_错误率(%)'] = f"{label_error_rate:.2f}"
        
        writer.writerow(total_row)
    
    print(f"\n统计完成:")
    print(f"- 源视频总数: {len(video_stats)}")
    print(f"- 样本总数: {total_samples}")
    print(f"- 总错误数: {total_errors}")
    print(f"- 总体错误率: {overall_error_rate:.2f}%")
    
    # 显示各类标签统计
    print(f"\n各类标签统计:")
    for label in label_categories:
        label_total = total_label_stats[label]['total']
        label_errors = total_label_stats[label]['errors']
        label_error_rate = (label_errors / label_total * 100) if label_total > 0 else 0
        print(f"- {label}: {label_total}个样本, {label_errors}个错误, 错误率{label_error_rate:.2f}%")
    
    print(f"\n- CSV报告已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='生成每个源视频错误次数的CSV统计')
    parser.add_argument('--dataset_path', 
                       default='LLaMA-Factory/data/subj_multi_test_data.json',
                       help='测试数据集路径')
    parser.add_argument('--predictions_path', 
                       default='LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-09-20-23-50-43/generated_predictions.jsonl',
                       help='预测结果文件路径')
    parser.add_argument('--output_path', 
                       default='video_error_stats.csv',
                       help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 测试数据集文件不存在: {args.dataset_path}")
        return
    
    if not os.path.exists(args.predictions_path):
        print(f"错误: 预测结果文件不存在: {args.predictions_path}")
        return
    
    generate_video_error_stats(args.dataset_path, args.predictions_path, args.output_path)

if __name__ == '__main__':
    main()