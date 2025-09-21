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

# 尝试导入 scikit-learn，如果失败则提供友好提示
try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: 未安装 scikit-learn，将跳过详细分类指标计算")
    print("安装命令: pip install scikit-learn")


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


def load_dataset_mapping(dataset_path: str) -> Dict[int, str]:
    """加载原始数据集信息，建立样本索引到视频文件的映射关系"""
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"警告: 数据集路径无效或不存在: {dataset_path}")
        return {}
    
    index_to_video = {}
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    # 假设数据集中包含 video_file 字段
                    video_file = data.get('video_file', '')
                    if video_file:
                        index_to_video[line_num] = video_file
                except json.JSONDecodeError:
                    print(f"警告: 数据集第{line_num + 1}行JSON格式错误")
                    continue
        
        print(f"📁 成功加载数据集映射: {len(index_to_video)} 个样本")
        
    except Exception as e:
        print(f"错误: 加载数据集失败 - {e}")
        return {}
    
    return index_to_video


def calculate_accuracy(predictions: List[Tuple[str, str]], dataset_mapping: Dict[int, str] = None) -> Dict:
    """计算准确度统计和详细分类指标，并记录错误样本信息"""
    if not predictions:
        return {"error": "没有有效的预测数据"}

    # 统计总体准确度
    correct = sum(1 for pred, label in predictions if pred == label)
    total = len(predictions)
    overall_accuracy = correct / total

    # 统计各类别准确度和错误样本
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion_matrix_dict = defaultdict(lambda: defaultdict(int))
    error_samples = []  # 记录错误样本信息

    for idx, (pred, label) in enumerate(predictions):
        class_stats[label]["total"] += 1
        confusion_matrix_dict[label][pred] += 1

        if pred == label:
            class_stats[label]["correct"] += 1
        else:
            # 记录错误样本信息
            error_info = {
                "sample_index": idx,
                "predicted_state": pred,
                "true_state": label,
                "video_file": dataset_mapping.get(idx, "未知") if dataset_mapping else "未提供数据集映射"
            }
            error_samples.append(error_info)

    # 计算各类别准确度
    class_accuracy = {}
    for class_name, stats in class_stats.items():
        class_accuracy[class_name] = stats["correct"] / stats["total"]

    # 统计预测分布
    pred_counter = Counter(pred for pred, _ in predictions)
    label_counter = Counter(label for _, label in predictions)

    # 基础结果
    results = {
        "overall_accuracy": overall_accuracy,
        "correct_predictions": correct,
        "total_predictions": total,
        "class_accuracy": class_accuracy,
        "class_stats": dict(class_stats),
        "confusion_matrix": dict(confusion_matrix_dict),
        "prediction_distribution": dict(pred_counter),
        "label_distribution": dict(label_counter),
        "error_samples": error_samples,
        "error_count": len(error_samples)
    }

    # 如果 scikit-learn 可用，计算详细分类指标
    if SKLEARN_AVAILABLE:
        # 准备数据
        y_true = [label for _, label in predictions]
        y_pred = [pred for pred, _ in predictions]

        # 定义标签顺序
        labels = ["饱腹", "中性", "微饿", "饥饿"]

        # 计算详细指标
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        # 计算宏平均和加权平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # 生成分类报告
        classification_rep = classification_report(
            y_true, y_pred, labels=labels, target_names=labels,
            output_dict=True, zero_division=0
        )

        # 添加详细指标到结果（转换numpy类型为Python原生类型）
        results.update({
            "detailed_metrics": {
                "per_class": {
                    labels[i]: {
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "f1_score": float(f1[i]),
                        "support": int(support[i])
                    } for i in range(len(labels))
                },
                "macro_avg": {
                    "precision": float(macro_precision),
                    "recall": float(macro_recall),
                    "f1_score": float(macro_f1)
                },
                "weighted_avg": {
                    "precision": float(weighted_precision),
                    "recall": float(weighted_recall),
                    "f1_score": float(weighted_f1)
                }
            },
            "classification_report": classification_rep
        })

    return results


def print_results(results: Dict):
    """打印结果"""
    if "error" in results:
        print(f"错误: {results['error']}")
        return

    print("=" * 80)
    print("新生儿喂养状态评估准确度统计")
    print("=" * 80)

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

    # 详细分类指标（如果可用）
    if "detailed_metrics" in results and SKLEARN_AVAILABLE:
        print_detailed_metrics(results)


def print_detailed_metrics(results: Dict):
    """打印详细的分类指标"""
    metrics = results["detailed_metrics"]

    print("\n" + "=" * 80)
    print("📊 详细分类指标")
    print("=" * 80)

    # 每个类别的详细指标
    print(f"\n📈 各类别详细指标:")
    print(f"{'类别':>6} {'精确度':>10} {'召回率':>10} {'F1分数':>10} {'支持度':>10}")
    print("-" * 60)

    states = ["饱腹", "中性", "微饿", "饥饿"]
    for state in states:
        if state in metrics["per_class"]:
            m = metrics["per_class"][state]
            print(f"{state:>6} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1_score']:>10.4f} {m['support']:>10.0f}")

    print("-" * 60)

    # 宏平均
    macro = metrics["macro_avg"]
    print(f"{'宏平均':>6} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
          f"{macro['f1_score']:>10.4f} {'':>10}")

    # 加权平均
    weighted = metrics["weighted_avg"]
    print(f"{'加权平均':>6} {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} "
          f"{weighted['f1_score']:>10.4f} {'':>10}")

    # 总体指标摘要
    print(f"\n🎯 总体指标摘要:")
    print(f"  总体准确度: {results['overall_accuracy']:.4f}")
    print(f"  宏平均 F1: {macro['f1_score']:.4f}")
    print(f"  加权平均 F1: {weighted['f1_score']:.4f}")

    # 性能分析
    print(f"\n📋 性能分析:")
    per_class = metrics["per_class"]

    # 找出表现最好和最差的类别
    f1_scores = {state: per_class[state]['f1_score'] for state in states if state in per_class}
    if f1_scores:
        best_class = max(f1_scores, key=f1_scores.get)
        worst_class = min(f1_scores, key=f1_scores.get)

        print(f"  表现最好类别: {best_class} (F1: {f1_scores[best_class]:.4f})")
        print(f"  表现最差类别: {worst_class} (F1: {f1_scores[worst_class]:.4f})")

        # 分析需要改进的类别
        low_f1_classes = [state for state, f1 in f1_scores.items() if f1 < 0.9]
        if low_f1_classes:
            print(f"  需要改进类别: {', '.join(low_f1_classes)}")
        else:
            print(f"  所有类别表现良好 (F1 > 0.9)")


def save_results(results: Dict, output_file: str):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file}")


def save_error_samples(error_samples: List[Dict], output_file: str):
    """保存错误样本信息到文件"""
    if not error_samples:
        print("没有错误样本需要保存")
        return
    
    # 保存为JSON格式
    error_output_file = output_file.replace('.json', '_error_samples.json')
    with open(error_output_file, 'w', encoding='utf-8') as f:
        json.dump(error_samples, f, ensure_ascii=False, indent=2)
    
    # 同时保存为易读的文本格式
    text_output_file = output_file.replace('.json', '_error_samples.txt')
    with open(text_output_file, 'w', encoding='utf-8') as f:
        f.write(f"错误样本详细信息 (共 {len(error_samples)} 个)\n")
        f.write("=" * 60 + "\n\n")
        
        for i, error in enumerate(error_samples, 1):
            f.write(f"错误样本 #{i}:\n")
            f.write(f"  样本索引: {error['sample_index']}\n")
            f.write(f"  预测状态: {error['predicted_state']}\n")
            f.write(f"  真实状态: {error['true_state']}\n")
            f.write(f"  源视频文件: {error['video_file']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"错误样本信息已保存到:")
    print(f"  JSON格式: {error_output_file}")
    print(f"  文本格式: {text_output_file}")
    print(f"  错误样本总数: {len(error_samples)}")


def main():
    parser = argparse.ArgumentParser(description="新生儿喂养状态评估准确度统计")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_file", type=str, 
                      help="直接指定generated_predictions.jsonl文件路径")
    group.add_argument("--eval_dir", type=str,
                      help="指定评估目录，自动查找generated_predictions.jsonl")
    
    parser.add_argument("--output_file", type=str, default=None,
                       help="保存详细结果的JSON文件路径（可选）")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="原始数据集路径，用于追踪错误样本的源视频文件")
    
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
    
    # 加载原始数据集映射（如果提供了数据集路径）
    dataset_mapping = None
    if args.dataset_path:
        print(f"📁 加载数据集映射: {args.dataset_path}")
        dataset_mapping = load_dataset_mapping(args.dataset_path)
    
    # 加载和分析数据
    predictions = load_predictions(input_file)
    results = calculate_accuracy(predictions, dataset_mapping)
    
    # 打印结果
    print_results(results)
    
    # 保存错误样本信息（如果有错误样本且提供了输出文件路径）
    if args.output_file and results.get('error_samples'):
        save_error_samples(results['error_samples'], args.output_file)
    
    # 保存结果（如果指定了输出文件）
    if args.output_file:
        save_results(results, args.output_file)


if __name__ == "__main__":
    main()