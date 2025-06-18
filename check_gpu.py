import torch
import subprocess
import os

def check_gpu_info():
    """检查GPU信息和显存使用情况"""
    print("=== GPU 信息检查 ===")
    
    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查CUDA安装")
        return
    
    # 检查CUDA版本
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查GPU数量和型号
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  总显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  多处理器数量: {props.multi_processor_count}")
        print(f"  CUDA计算能力: {props.major}.{props.minor}")
    
    # 检查当前显存使用情况
    print("\n=== 显存使用情况 ===")
    for i in range(gpu_count):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}:")
        print(f"  已分配显存: {allocated:.2f} GB")
        print(f"  缓存显存: {cached:.2f} GB") 
        print(f"  总显存: {total:.1f} GB")
        print(f"  使用率: {(allocated/total)*100:.1f}%")
    
    # 检查环境变量
    print("\n=== 环境变量 ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"当前设备: {torch.cuda.current_device()}")
    
    # 使用nvidia-smi获取详细信息
    print("\n=== nvidia-smi 信息 ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"无法运行nvidia-smi: {e}")

def estimate_memory_requirement():
    """估算模型所需显存"""
    print("\n=== Qwen2.5-VL-7B 显存需求估算 ===")
    
    # 模型参数量约7B
    model_params = 7e9
    
    # 不同精度下的显存需求
    fp32_size = model_params * 4 / 1024**3  # 4 bytes per param
    fp16_size = model_params * 2 / 1024**3  # 2 bytes per param
    int8_size = model_params * 1 / 1024**3  # 1 byte per param
    
    print(f"模型参数: {model_params/1e9:.1f}B")
    print(f"FP32模型大小: {fp32_size:.1f} GB")
    print(f"FP16模型大小: {fp16_size:.1f} GB") 
    print(f"INT8模型大小: {int8_size:.1f} GB")
    
    # 训练时还需要额外显存用于：
    # - 梯度 (与模型大小相同)
    # - 优化器状态 (Adam约为模型大小的2倍)
    # - 激活值缓存
    # - 批处理数据
    
    training_overhead = 3  # 保守估计为模型大小的3倍
    
    print(f"\n训练显存估算:")
    print(f"FP32训练: {fp32_size * (1 + training_overhead):.1f} GB")
    print(f"FP16训练: {fp16_size * (1 + training_overhead):.1f} GB")
    print(f"推荐显存: 24GB以上 (RTX 3090/4090/A100)")

if __name__ == "__main__":
    check_gpu_info()
    estimate_memory_requirement() 