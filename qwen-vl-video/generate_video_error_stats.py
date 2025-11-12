#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成每个源视频错误次数的CSV统计脚本

功能：
1. 读取测试数据集和预测结果文件
2. 从视频路径中提取源视频文件名
3. 按源视频分组统计错误率
4. 生成CSV格式的统计报告，包含各类标签的详细统计
5. 分析错误样本中的文本动作统计情况
"""

import json
import csv
import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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

def parse_action_statistics(content: str) -> Dict[str, float]:
    """解析文本中的行为观察统计信息
    
    从样本的content中提取行为统计数据，包括：
    - 吸吮行为次数和时长
    - 手脚活动次数和时长
    - 皱眉次数和时长
    - 总行为次数和总持续时间
    """
    action_stats = {
        'sucking_count': 0.0,
        'sucking_duration': 0.0,
        'limb_activity_count': 0.0,
        'limb_activity_duration': 0.0,
        'frowning_count': 0.0,
        'frowning_duration': 0.0,
        'total_count': 0.0,
        'total_duration': 0.0
    }
    
    # 解析吸吮行为
    sucking_pattern = r'吸吮行为\((\d+)次,\s*([\d.]+)秒\)'
    sucking_match = re.search(sucking_pattern, content)
    if sucking_match:
        action_stats['sucking_count'] = float(sucking_match.group(1))
        action_stats['sucking_duration'] = float(sucking_match.group(2))
    
    # 解析手脚活动
    limb_pattern = r'手脚活动加快\((\d+)次,\s*([\d.]+)秒\)'
    limb_match = re.search(limb_pattern, content)
    if limb_match:
        action_stats['limb_activity_count'] = float(limb_match.group(1))
        action_stats['limb_activity_duration'] = float(limb_match.group(2))
    
    # 解析皱眉
    frowning_pattern = r'皱眉\((\d+)次,\s*([\d.]+)秒\)'
    frowning_match = re.search(frowning_pattern, content)
    if frowning_match:
        action_stats['frowning_count'] = float(frowning_match.group(1))
        action_stats['frowning_duration'] = float(frowning_match.group(2))
    
    # 解析总计信息
    total_pattern = r'总计观察到(\d+)次行为活动，持续时间([\d.]+)秒'
    total_match = re.search(total_pattern, content)
    if total_match:
        action_stats['total_count'] = float(total_match.group(1))
        action_stats['total_duration'] = float(total_match.group(2))
    
    return action_stats

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

def generate_action_analysis_csv(error_samples: List[Dict], output_path: str):
    """生成错误样本的行为分析CSV文件"""
    print(f"生成行为分析CSV报告: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            '样本索引', '源视频文件名', '预测标签', '真实标签',
            '吸吮行为次数', '吸吮行为时长(秒)', 
            '手脚活动次数', '手脚活动时长(秒)',
            '皱眉次数', '皱眉时长(秒)',
            '总行为次数', '总持续时间(秒)'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample in error_samples:
            action_stats = sample['action_stats']
            row_data = {
                '样本索引': sample['index'],
                '源视频文件名': sample['source_video'],
                '预测标签': sample['pred_label'],
                '真实标签': sample['true_label'],
                '吸吮行为次数': int(action_stats['sucking_count']),
                '吸吮行为时长(秒)': action_stats['sucking_duration'],
                '手脚活动次数': int(action_stats['limb_activity_count']),
                '手脚活动时长(秒)': action_stats['limb_activity_duration'],
                '皱眉次数': int(action_stats['frowning_count']),
                '皱眉时长(秒)': action_stats['frowning_duration'],
                '总行为次数': int(action_stats['total_count']),
                '总持续时间(秒)': action_stats['total_duration']
            }
            writer.writerow(row_data)
    
    print(f"- 行为分析CSV报告已保存到: {output_path}")

def analyze_action_differences(correct_samples: List[Dict], error_samples: List[Dict]):
    """分析错误样本与正确样本在行为特征上的差异"""
    def calculate_avg_stats(samples: List[Dict]) -> Dict[str, float]:
        if not samples:
            return {}
        
        total_stats = {
            'sucking_count': 0.0, 'sucking_duration': 0.0,
            'limb_activity_count': 0.0, 'limb_activity_duration': 0.0,
            'frowning_count': 0.0, 'frowning_duration': 0.0,
            'total_count': 0.0, 'total_duration': 0.0
        }
        
        for sample in samples:
            action_stats = sample['action_stats']
            for key in total_stats:
                total_stats[key] += action_stats[key]
        
        # 计算平均值
        avg_stats = {key: value / len(samples) for key, value in total_stats.items()}
        return avg_stats
    
    correct_avg = calculate_avg_stats(correct_samples)
    error_avg = calculate_avg_stats(error_samples)
    
    print(f"\n行为特征差异分析:")
    print(f"正确样本数量: {len(correct_samples)}, 错误样本数量: {len(error_samples)}")
    print(f"\n平均行为统计对比:")
    
    behavior_names = {
        'sucking_count': '吸吮行为次数',
        'sucking_duration': '吸吮行为时长(秒)',
        'limb_activity_count': '手脚活动次数',
        'limb_activity_duration': '手脚活动时长(秒)',
        'frowning_count': '皱眉次数',
        'frowning_duration': '皱眉时长(秒)',
        'total_count': '总行为次数',
        'total_duration': '总持续时间(秒)'
    }
    
    for key, name in behavior_names.items():
        correct_val = correct_avg.get(key, 0)
        error_val = error_avg.get(key, 0)
        diff = error_val - correct_val
        diff_pct = (diff / correct_val * 100) if correct_val > 0 else 0
        print(f"- {name}: 正确样本={correct_val:.2f}, 错误样本={error_val:.2f}, 差异={diff:+.2f} ({diff_pct:+.1f}%)")

def generate_video_error_stats(dataset_path: str, predictions_path: str, output_path: str, action_analysis_output: Optional[str] = None):
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
    
    # 收集错误样本和正确样本用于行为分析
    error_samples = []
    correct_samples = []
    
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
        
        # 解析行为统计信息
        content = sample['messages'][0]['content']
        action_stats = parse_action_statistics(content)
        
        # 判断是否为错误样本
        is_error = true_label != pred_label
        
        # 统计总体
        video_stats[source_video]['total'] += 1
        video_stats[source_video]['samples'].append({
            'index': i,
            'video_path': video_path,
            'true_label': true_label,
            'pred_label': pred_label,
            'is_error': is_error
        })
        
        if is_error:
            video_stats[source_video]['errors'] += 1
            # 收集错误样本用于行为分析
            error_samples.append({
                'index': i,
                'source_video': source_video,
                'pred_label': pred_label,
                'true_label': true_label,
                'action_stats': action_stats
            })
        else:
            # 收集正确样本用于行为分析
            correct_samples.append({
                'index': i,
                'source_video': source_video,
                'pred_label': pred_label,
                'true_label': true_label,
                'action_stats': action_stats
            })
        
        # 统计各类标签
        if true_label in label_categories:
            video_stats[source_video]['label_stats'][true_label]['total'] += 1
            if is_error:
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
    
    # 生成行为分析报告
    if action_analysis_output:
        generate_action_analysis_csv(error_samples, action_analysis_output)
        analyze_action_differences(correct_samples, error_samples)

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
    parser.add_argument('--action_analysis_output',
                       help='行为分析输出CSV文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 测试数据集文件不存在: {args.dataset_path}")
        return
    
    if not os.path.exists(args.predictions_path):
        print(f"错误: 预测结果文件不存在: {args.predictions_path}")
        return
    
    generate_video_error_stats(args.dataset_path, args.predictions_path, args.output_path, args.action_analysis_output)

if __name__ == '__main__':
    main()