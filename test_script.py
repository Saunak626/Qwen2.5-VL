#!/usr/bin/env python3
import os
import sys

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("\nChecking file paths:")

# 检查数据文件
dataset_path = 'LLaMA-Factory/data/subj_multi_test_data.json'
print(f"Dataset path: {dataset_path}")
print(f"Dataset exists: {os.path.exists(dataset_path)}")

# 检查预测文件
predictions_path = 'LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-09-20-23-50-43/generated_predictions.jsonl'
print(f"Predictions path: {predictions_path}")
print(f"Predictions exists: {os.path.exists(predictions_path)}")

# 检查脚本文件
script_path = 'qwen-vl-video/generate_video_error_stats.py'
print(f"Script path: {script_path}")
print(f"Script exists: {os.path.exists(script_path)}")

# 列出当前目录
print("\nCurrent directory contents:")
for item in os.listdir('.'):
    print(f"  {item}")

# 检查LLaMA-Factory目录
if os.path.exists('LLaMA-Factory'):
    print("\nLLaMA-Factory directory contents:")
    for item in os.listdir('LLaMA-Factory'):
        print(f"  {item}")
    
    if os.path.exists('LLaMA-Factory/data'):
        print("\nLLaMA-Factory/data directory contents:")
        for item in os.listdir('LLaMA-Factory/data'):
            print(f"  {item}")
else:
    print("\nLLaMA-Factory directory does not exist")

print("\nTest completed.")