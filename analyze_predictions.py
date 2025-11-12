#!/usr/bin/env python3
"""
模型预测结果准确性分析脚本

功能:
1. 读取预测结果和数据集文件
2. 验证数据对齐
3. 按视频分组统计准确率
4. 输出详细的统计报告
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_video_identifier(video_path: str) -> str:
    """
    从视频路径中提取视频标识符
    
    示例:
    输入: /home/swq/Code/.../video_clips/240717(9.03.26）_clip_300.mp4
    输出: 240717(9.03.26）
    """
    # 提取文件名部分
    filename = Path(video_path).name
    
    # 使用正则表达式提取日期部分 (格式: YYMMDD(H.MM.SS）)
    # 匹配模式: 数字开头，包含括号和时间信息，到 _clip_ 之前
    match = re.match(r'(.+?)_clip_\d+\.mp4', filename)
    
    if match:
        return match.group(1)
    else:
        # 如果无法匹配，返回去掉扩展名的文件名
        return filename.replace('.mp4', '')


def load_predictions(predictions_path: str) -> List[Dict]:
    """
    加载预测结果文件 (JSONL格式)
    """
    predictions = []
    
    if not Path(predictions_path).exists():
        raise FileNotFoundError(f"预测结果文件不存在: {predictions_path}")
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                predictions.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
    
    return predictions


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    加载数据集文件 (JSON格式)
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        try:
            dataset = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"数据集JSON解析失败: {e}")
    
    return dataset


def normalize_label(label: str) -> str:
    """
    标准化标签文本 (去除空白字符)
    """
    return label.strip()


def analyze_predictions(predictions: List[Dict], dataset: List[Dict]) -> Dict:
    """
    分析预测结果
    
    返回:
    {
        'video_id': {
            'total': int,
            'correct': int,
            'incorrect': int,
            'accuracy': float
        },
        ...
    }
    """
    # 验证样本数量
    if len(predictions) != len(dataset):
        print(f"警告: 预测结果数量 ({len(predictions)}) 与数据集数量 ({len(dataset)}) 不一致!")
        min_len = min(len(predictions), len(dataset))
        print(f"将只分析前 {min_len} 个样本")
        predictions = predictions[:min_len]
        dataset = dataset[:min_len]
    
    # 按视频分组统计
    video_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})
    
    # 用于验证数据对齐
    misalignment_count = 0
    
    for idx, (pred, data) in enumerate(zip(predictions, dataset)):
        # 提取预测结果和真实标签
        predicted_label = normalize_label(pred.get('predict', ''))
        true_label_from_pred = normalize_label(pred.get('label', ''))
        
        # 从数据集中提取真实标签
        try:
            true_label_from_data = normalize_label(data['messages'][1]['content'])
        except (KeyError, IndexError) as e:
            print(f"警告: 样本 {idx} 数据集标签提取失败: {e}")
            continue
        
        # 验证标签一致性
        if true_label_from_pred != true_label_from_data:
            misalignment_count += 1
            if misalignment_count <= 5:  # 只显示前5个不一致的样本
                print(f"警告: 样本 {idx} 标签不一致 - 预测文件: '{true_label_from_pred}', 数据集: '{true_label_from_data}'")
        
        # 提取视频路径和标识符
        try:
            video_path = data['videos'][0]
            video_id = extract_video_identifier(video_path)
        except (KeyError, IndexError) as e:
            print(f"警告: 样本 {idx} 视频路径提取失败: {e}")
            continue
        
        # 统计
        video_stats[video_id]['total'] += 1
        
        # 使用数据集中的真实标签进行对比
        if predicted_label == true_label_from_data:
            video_stats[video_id]['correct'] += 1
        else:
            video_stats[video_id]['incorrect'] += 1
    
    if misalignment_count > 0:
        print(f"\n总计发现 {misalignment_count} 个样本的标签不一致\n")
    
    # 计算准确率
    for video_id in video_stats:
        stats = video_stats[video_id]
        stats['accuracy'] = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
    
    return dict(video_stats)


def print_statistics(video_stats: Dict):
    """
    打印统计结果
    """
    print("=" * 80)
    print("视频级别预测准确性分析")
    print("=" * 80)
    print()
    
    # 按视频ID排序
    sorted_videos = sorted(video_stats.items(), key=lambda x: x[0])
    
    total_samples = 0
    total_correct = 0
    total_incorrect = 0
    
    for video_id, stats in sorted_videos:
        print(f"视频: {video_id}")
        print(f"  总样本数: {stats['total']}")
        print(f"  正确数: {stats['correct']}")
        print(f"  错误数: {stats['incorrect']}")
        print(f"  准确率: {stats['accuracy']:.2f}%")
        print()
        
        total_samples += stats['total']
        total_correct += stats['correct']
        total_incorrect += stats['incorrect']
    
    # 整体统计
    print("=" * 80)
    print("整体统计")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"总正确数: {total_correct}")
    print(f"总错误数: {total_incorrect}")
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    print(f"整体准确率: {overall_accuracy:.2f}%")
    print(f"视频总数: {len(video_stats)}")
    print("=" * 80)


def save_to_csv(video_stats: Dict, output_path: str):
    """
    将统计结果保存为CSV文件
    """
    import csv
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['视频ID', '总样本数', '正确数', '错误数', '准确率(%)'])
        
        # 按视频ID排序
        sorted_videos = sorted(video_stats.items(), key=lambda x: x[0])
        
        for video_id, stats in sorted_videos:
            writer.writerow([
                video_id,
                stats['total'],
                stats['correct'],
                stats['incorrect'],
                f"{stats['accuracy']:.2f}"
            ])
        
        # 添加整体统计
        total_samples = sum(s['total'] for s in video_stats.values())
        total_correct = sum(s['correct'] for s in video_stats.values())
        total_incorrect = sum(s['incorrect'] for s in video_stats.values())
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
        
        writer.writerow([])
        writer.writerow(['整体统计', total_samples, total_correct, total_incorrect, f"{overall_accuracy:.2f}"])
    
    print(f"\n统计结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='分析模型预测结果的准确性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认路径
  python analyze_predictions.py
  
  # 指定自定义路径
  python analyze_predictions.py --predictions path/to/predictions.jsonl --dataset path/to/dataset.json
  
  # 保存结果到CSV
  python analyze_predictions.py --output results.csv
        """
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        default='LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-09-20-23-50-43/generated_predictions.jsonl',
        help='预测结果文件路径 (JSONL格式)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='LLaMA-Factory/data/subj_multi_test_data.json',
        help='数据集文件路径 (JSON格式)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出CSV文件路径 (可选)'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print("正在加载预测结果...")
        predictions = load_predictions(args.predictions)
        print(f"已加载 {len(predictions)} 条预测结果")
        
        print("正在加载数据集...")
        dataset = load_dataset(args.dataset)
        print(f"已加载 {len(dataset)} 个数据样本")
        print()
        
        # 分析预测结果
        print("正在分析预测结果...")
        video_stats = analyze_predictions(predictions, dataset)
        print()
        
        # 打印统计结果
        print_statistics(video_stats)
        
        # 保存到CSV (如果指定)
        if args.output:
            save_to_csv(video_stats, args.output)
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

