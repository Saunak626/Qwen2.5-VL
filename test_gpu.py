import os

# 设置使用特定GPU (必须在导入torch之前设置)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用GPU 3

import torch
import subprocess

def test_gpu_assignment():
    """测试GPU分配是否正确"""
    print("=== GPU分配测试 ===")
    
    # 显示环境变量
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    
    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 显示可见的GPU数量（应该只有1个）
        device_count = torch.cuda.device_count()
        print(f"可见GPU数量: {device_count}")
        
        # 显示当前默认设备
        current_device = torch.cuda.current_device()
        print(f"当前设备: cuda:{current_device}")
        
        # 显示每个可见GPU的信息
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"cuda:{i} -> {props.name}")
            print(f"  总显存: {props.total_memory / 1024**3:.1f} GB")
        
        # 创建一个张量并显示它在哪个设备上
        test_tensor = torch.randn(10, 10).cuda()
        print(f"测试张量位置: {test_tensor.device}")
        
        # 显示显存使用情况
        for i in range(device_count):
            allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            print(f"cuda:{i} 已分配显存: {allocated:.1f} MB")
    
    print("\n=== nvidia-smi输出 ===")
    try:
        # 运行nvidia-smi查看实际GPU使用情况
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"无法运行nvidia-smi: {e}")

if __name__ == "__main__":
    test_gpu_assignment() 