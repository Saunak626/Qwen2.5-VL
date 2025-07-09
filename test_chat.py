import torch
import os

# GPU 指定方法（任选其一）
# 方法1：使用环境变量指定可见GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 仅使用GPU 1

# 方法2：在模型加载时指定设备映射
# device_map = {"": 1}  # 将模型加载到GPU 1
device_map = {"": "cuda:3"}  # 将模型加载到GPU 3（推荐，最空闲）

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# 从本地路径加载模型

# 推荐启用flash_attention_2以获得更好的加速和内存节省，特别是在多图像和视频场景中
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.

# 注意：这里使用flash_attention_2可能导致兼容性问题，如遇到错误可改为注释掉attn_implementation参数
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/swq/Code/Qwen2.5-VL/models/Qwen/Qwen2.5-VL-7B-Instruct",  # 模型本地路径
    torch_dtype=torch.bfloat16,  # 使用bfloat16数据类型以节省显存
    attn_implementation="flash_attention_2",
    device_map=device_map,  # 使用指定的GPU设备
    # device_map={"": 1},  # 指定使用GPU 1（取消注释以启用）
    # device_map={"": "cuda:3"},  # 指定使用GPU 3（取消注释以启用）
)

# 加载自动处理器，用于处理文本和图像输入
processor = AutoProcessor.from_pretrained("/home/swq/Code/Qwen2.5-VL/models/Qwen/Qwen2.5-VL-7B-Instruct")

# 模型中每张图像的视觉token数量默认范围是4-16384
# 可以根据需要设置min_pixels和max_pixels，例如token范围256-1280，以平衡性能和成本

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28  # 最小像素数设置
# max_pixels = 1280*28*28  # 最大像素数设置
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# 构建对话消息，包含用户角色和内容（图像+文本）
messages = [
    {
        "role": "user",  # 用户角色
        "content": [  # 消息内容列表
            # {
            #     "type": "image",  # 内容类型：图像
            #     "image": "coco_2014_caption/307.jpg",  # 本地图像路径（注释掉）
            #     # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",  # 在线图像URL
            # },
            {
                "type": "video",  # 内容类型
                "video": "Data/videos/2025-02-07 081750/2.mp4",  # 本地路径（注释掉）
            },
            {"type": "text", "text": "视频中人在做什么？他的行为动作有哪些？"},  # 内容类型：文本，询问图像内容
        ],
    }
]

# 推理准备：应用聊天模板生成文本提示
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True  # 应用模板但不分词，添加生成提示
)
# 处理视觉信息，提取图像和视频输入
image_inputs, video_inputs = process_vision_info(messages)

# 使用处理器处理所有输入（文本、图像、视频）
inputs = processor(
    text=[text],          # 文本输入列表
    images=image_inputs,  # 图像输入
    videos=video_inputs,  # 视频输入
    padding=True,         # 启用填充
    return_tensors="pt",  # 返回PyTorch张量
)

# 将输入张量移动到指定的GPU设备
inputs = inputs.to("cuda:3")  # 移动到GPU 3

# 推理：生成输出文本
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128) # 生成最多128个新token，150为备选值

# 裁剪生成的token，移除输入部分，只保留新生成的内容
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
# 将生成的token解码为文本
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
# 打印输出结果
print(output_text)
