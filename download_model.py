# 模型下载 （这一步需要记录下载的位置）
from modelscope import snapshot_download

# 自定义下载路径
custom_path = '/home/swq/Code/Qwen2.5-VL/models'

# 下载模型并指定自定义路径
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir=custom_path)

print(f"Model downloaded to: {model_dir}")