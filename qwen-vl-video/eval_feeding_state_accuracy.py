#!/usr/bin/env python3
"""
新生儿喂养状态评估准确度统计脚本

用法:
python eval_feeding_state_accuracy.py --input_file /path/to/generated_predictions.jsonl
python eval_feeding_state_accuracy.py --eval_dir LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-09-01-10-28-39
"""

import json
import argparse
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def parse_state_from_json_string(data) -> str:
    """从JSON字符串或直接字符串中解析state字段"""
    if isinstance(data, str):
        # 如果是字符串，先尝试作为JSON解析
        try:
            parsed = json.loads(data)
            return parsed.get("state", "").strip()
        except (json.JSONDecodeError, TypeError):
            # 如果JSON解析失败，直接返回清理后的字符串
            return data.strip()
    elif isinstance(data, dict):
        # 如果是字典，直接获取state字段
        return data.get("state", "").strip()
    else:
        # 其他类型返回空字符串
        return ""


def load_predictions(file_path: str) -> List[Tuple[str, str]]:
    """加载预测结果，返回(predict_state, label_state)的列表"""
    predictions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                predict_state = parse_state_from_json_string(data.get("predict", ""))
                label_state = parse_state_from_json_string(data.get("label", ""))
                
                if predict_state and label_state:
                    predictions.append((predict_state, label_state))
                else:
                    print(f"警告: 第{line_num}行数据解析失败")
                    
            except json.JSONDecodeError:
                print(f"警告: 第{line_num}行JSON格式错误")
                continue
    
    return predictions


def calculate_accuracy(predictions: List[Tuple[str, str]]) -> Dict:
    """计算准确度统计"""
    if not predictions:
        return {"error": "没有有效的预测数据"}
    
    # 统计总体准确度
    correct = sum(1 for pred, label in predictions if pred == label)
    total = len(predictions)
    overall_accuracy = correct / total
    
    # 统计各类别准确度
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for pred, label in predictions:
        class_stats[label]["total"] += 1
        confusion_matrix[label][pred] += 1
        
        if pred == label:
            class_stats[label]["correct"] += 1
    
    # 计算各类别准确度
    class_accuracy = {}
    for class_name, stats in class_stats.items():
        class_accuracy[class_name] = stats["correct"] / stats["total"]
    
    # 统计预测分布
    pred_counter = Counter(pred for pred, _ in predictions)
    label_counter = Counter(label for _, label in predictions)
    
    return {
        "overall_accuracy": overall_accuracy,
        "correct_predictions": correct,
        "total_predictions": total,
        "class_accuracy": class_accuracy,
        "class_stats": dict(class_stats),
        "confusion_matrix": dict(confusion_matrix),
        "prediction_distribution": dict(pred_counter),
        "label_distribution": dict(label_counter)
    }


def print_results(results: Dict):
    """打印结果"""
    if "error" in results:
        print(f"错误: {results['error']}")
        return
    
    print("=" * 60)
    print("新生儿喂养状态评估准确度统计")
    print("=" * 60)
    
    # 总体准确度
    print(f"\n📊 总体准确度: {results['overall_accuracy']:.4f} "
          f"({results['correct_predictions']}/{results['total_predictions']})")
    
    # 各类别准确度
    print(f"\n📈 各类别准确度:")
    states = ["饱腹", "中性", "微饿", "饥饿"]
    for state in states:
        if state in results['class_accuracy']:
            acc = results['class_accuracy'][state]
            stats = results['class_stats'][state]
            print(f"  {state:>4}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # 混淆矩阵
    print(f"\n🔄 混淆矩阵 (真实 → 预测):")
    print(f"{'':>6}", end="")
    for state in states:
        print(f"{state:>6}", end="")
    print()
    
    for true_state in states:
        if true_state in results['confusion_matrix']:
            print(f"{true_state:>6}", end="")
            for pred_state in states:
                count = results['confusion_matrix'][true_state].get(pred_state, 0)
                print(f"{count:>6}", end="")
            print()
    
    # 分布统计
    print(f"\n📋 标签分布:")
    for state in states:
        count = results['label_distribution'].get(state, 0)
        ratio = count / results['total_predictions']
        print(f"  {state:>4}: {count:>4} ({ratio:.2%})")
    
    print(f"\n📋 预测分布:")
    for state in states:
        count = results['prediction_distribution'].get(state, 0)
        ratio = count / results['total_predictions']
        print(f"  {state:>4}: {count:>4} ({ratio:.2%})")


def save_results(results: Dict, output_file: str):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="新生儿喂养状态评估准确度统计")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_file", type=str, 
                      help="直接指定generated_predictions.jsonl文件路径")
    group.add_argument("--eval_dir", type=str,
                      help="指定评估目录，自动查找generated_predictions.jsonl")
    
    parser.add_argument("--output_file", type=str, default=None,
                       help="保存详细结果的JSON文件路径（可选）")
    
    args = parser.parse_args()
    
    # 确定输入文件路径
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = os.path.join(args.eval_dir, "generated_predictions.jsonl")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        return
    
    print(f"📁 读取文件: {input_file}")
    
    # 加载和分析数据
    predictions = load_predictions(input_file)
    results = calculate_accuracy(predictions)
    
    # 打印结果
    print_results(results)
    
    # 保存结果（如果指定了输出文件）
    if args.output_file:
        save_results(results, args.output_file)


if __name__ == "__main__":
    main()