"""
Qwen2.5-VL 视觉语言模型 LoRA 微调训练脚本

此脚本用于在 COCO 数据集上微调 Qwen2.5-VL-7B 模型，使用 LoRA (Low-Rank Adaptation) 技术
实现高效的参数更新，仅训练少量参数即可获得良好的微调效果。

主要功能：
1. 加载预训练的 Qwen2.5-VL-7B 模型
2. 配置 LoRA 参数进行高效微调
3. 处理多模态数据（图像+文本）
4. 执行训练并保存检查点
5. 在测试集上进行推理验证

技术要点：
- 使用 FP32 精度避免数值不稳定
- 梯度检查点减少显存占用
- SwanLab 进行训练监控和可视化
"""

import os

# ================================ GPU 设备配置 ================================
# 设置使用特定GPU (必须在导入torch之前设置，否则无效)
# CUDA_VISIBLE_DEVICES 环境变量告诉 CUDA 运行时只使用指定的物理GPU
# 设置为 "3" 表示只使用物理编号为3的GPU，在程序中这个GPU会被映射为 cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用GPU 3

# ================================ 导入依赖库 ================================
import torch                                    # PyTorch 深度学习框架
from datasets import Dataset                    # HuggingFace datasets 库，用于数据处理
from modelscope import snapshot_download, AutoTokenizer    # ModelScope 模型下载和分词器
from swanlab.integration.transformers import SwanLabCallback  # SwanLab 训练监控
from qwen_vl_utils import process_vision_info   # Qwen-VL 视觉信息处理工具
from peft import LoraConfig, TaskType, get_peft_model, PeftModel  # PEFT 库用于 LoRA 微调
from transformers import (
    TrainingArguments,                          # 训练参数配置
    Trainer,                                    # 训练器
    DataCollatorForSeq2Seq,                     # 序列到序列数据整理器
    Qwen2_5_VLForConditionalGeneration,         # Qwen2.5-VL 模型
    AutoProcessor,                              # 自动处理器（文本+图像）
)
import swanlab                                  # 实验跟踪和可视化
import json                                     # JSON 数据处理

# ================================ GPU 信息显示 ================================
# 验证GPU配置是否正确，显示当前可用的GPU设备信息
print(f"使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")  # 显示环境变量设置
print(f"CUDA可用: {torch.cuda.is_available()}")              # 检查CUDA是否可用
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")          # 显示可见GPU数量（应该为1）
    # 遍历所有可见GPU，显示设备信息和显存容量
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")  # GPU名称
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  总显存: {total_memory:.1f} GB")            # 显存大小（GB）

# ================================ 数据预处理函数 ================================
def process_func(example):
    """
    对训练数据集中的每一条样本进行预处理，转换为模型可以接受的格式
    
    处理流程：
    1. 解析对话数据，提取用户输入（图像+文本）和模型回答
    2. 构建多模态消息格式（图像+文本提示）
    3. 使用 processor 处理图像和文本，生成 token
    4. 将输入和输出拼接，创建训练序列
    5. 设置标签（labels），其中输入部分设为-100（不计算损失），输出部分计算损失
    
    Args:
        example: 包含 "conversations" 字段的数据样本
        
    Returns:
        dict: 包含 input_ids, attention_mask, labels, pixel_values, image_grid_thw 的训练样本
    """
    MAX_LENGTH = 8192                                        # 最大序列长度限制
    input_ids, attention_mask, labels = [], [], []          # 初始化返回变量
    
    # 解析对话数据结构
    conversation = example["conversations"]                  # 获取对话内容
    input_content = conversation[0]["value"]                 # 用户输入（包含图像路径和文本）
    output_content = conversation[1]["value"]                # 模型应该生成的回答
    
    # 从输入内容中提取图像文件路径
    # 输入格式：文本<|vision_start|>图像路径<|vision_end|>文本
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 截取图像路径
    # 构建多模态消息格式，符合 Qwen-VL 的输入要求
    # 每个消息包含角色和内容，内容可以是图像和文本的组合
    messages = [
        {
            "role": "user",                                      # 用户角色
            "content": [                                         # 内容列表，支持多模态
                {
                    "type": "image",                             # 图像类型
                    "image": f"{file_path}",                     # 图像文件路径
                    "resized_height": 280,                       # 图像调整后的高度
                    "resized_width": 280,                        # 图像调整后的宽度
                },
                {"type": "text", "text": "COCO Yes:"},           # 文本提示，引导模型生成描述
            ],
        }
    ]
    
    # 使用 processor 将消息转换为聊天模板格式的文本
    # apply_chat_template: 将对话格式转换为模型期望的模板格式
    # tokenize=False: 不进行分词，返回原始文本
    # add_generation_prompt=True: 添加生成提示符，表示模型应该开始生成回答
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取格式化的文本
    
    # 处理视觉信息，提取图像和视频数据
    # process_vision_info: Qwen-VL 专用函数，处理消息中的图像和视频
    # 返回预处理后的图像张量和视频张量（如果有）
    image_inputs, video_inputs = process_vision_info(messages)  # 获取预处理的视觉数据
    
    # 使用 processor 同时处理文本和视觉数据
    # processor 是 AutoProcessor，能够同时处理文本分词和图像编码
    inputs = processor(
        text=[text],                                             # 文本数据（列表格式）
        images=image_inputs,                                     # 图像数据
        videos=video_inputs,                                     # 视频数据（本例中为空）
        padding=True,                                            # 填充到批次中最长序列的长度
        return_tensors="pt",                                     # 返回 PyTorch 张量格式
    )
    
    # 将张量转换为列表格式，便于后续的序列拼接操作
    # 因为需要将输入序列和输出序列拼接成完整的训练序列
    inputs = {key: value.tolist() for key, value in inputs.items()} 
    
    # 分离指令部分和回答部分，为损失计算做准备
    instruction = inputs                                         # 输入指令部分（图像+文本提示）
    # 对期望的输出内容进行分词，不添加特殊token
    response = tokenizer(f"{output_content}", add_special_tokens=False)  # 目标回答部分

    # 构建完整的训练序列：[输入序列] + [输出序列] + [结束符]
    # 这样模型可以学习在给定输入的情况下生成正确的输出
    input_ids = (
            instruction["input_ids"][0] +                        # 输入部分的 token IDs
            response["input_ids"] +                              # 期望输出的 token IDs  
            [tokenizer.eos_token_id]                             # 序列结束符
    )

    # 构建注意力掩码，标记哪些位置是有效的token（1表示有效，0表示填充）
    attention_mask = (
        instruction["attention_mask"][0] +                       # 输入部分的注意力掩码
        response["attention_mask"] +                             # 输出部分的注意力掩码
        [1]                                                      # 结束符的注意力掩码
    )
    
    # 构建标签序列，用于计算损失
    # -100 是 PyTorch 交叉熵损失函数的忽略索引，表示不计算该位置的损失
    # 只有输出部分（模型应该生成的内容）才计算损失
    labels = (
            [-100] * len(instruction["input_ids"][0]) +          # 输入部分设为-100，不计算损失
            response["input_ids"] +                              # 输出部分计算损失
            [tokenizer.eos_token_id]                             # 结束符也要计算损失
    )
    
    # 序列长度截断，防止超出模型最大长度限制
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]                       # 截断输入序列
        attention_mask = attention_mask[:MAX_LENGTH]             # 截断注意力掩码
        labels = labels[:MAX_LENGTH]                             # 截断标签序列

    # 转换为 PyTorch 张量，指定合适的数据类型
    input_ids = torch.tensor(input_ids, dtype=torch.long)       # token IDs（整数类型）
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)  # 注意力掩码（整数类型）
    
    labels = torch.tensor(labels, dtype=torch.long)             # 标签（整数类型）
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'], dtype=torch.float32)  # 图像像素值（浮点类型）
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 图像网格尺寸，去除批次维度  (1,h,w)->(h,w)
    
    # 返回处理后的训练样本，包含模型训练所需的所有输入
    return {
        "input_ids": input_ids,                                  # 输入token序列
        "attention_mask": attention_mask,                        # 注意力掩码
        "labels": labels,                                        # 训练标签
        "pixel_values": inputs['pixel_values'],                 # 图像像素数据
        "image_grid_thw": inputs['image_grid_thw']               # 图像网格信息
    }


# ================================ 推理预测函数 ================================
def predict(messages, model):
    """
    使用训练好的模型进行推理预测
    
    此函数接收多模态消息（图像+文本），通过模型生成相应的文本回答
    主要用于测试阶段验证模型的性能
    
    Args:
        messages: 多模态消息列表，包含图像和文本
        model: 训练好的 LoRA 模型
        
    Returns:
        str: 模型生成的文本回答
    """
    # 使用与训练时相同的方式处理输入消息
    # 将多模态消息转换为模型可以理解的格式
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True     # 添加生成提示，告诉模型开始生成
    )
    
    # 提取和预处理视觉信息（图像）
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 使用 processor 同时处理文本和图像，生成模型输入
    inputs = processor(
        text=[text],                                             # 文本输入
        images=image_inputs,                                     # 图像输入
        videos=video_inputs,                                     # 视频输入（通常为空）
        padding=True,                                            # 填充序列
        return_tensors="pt",                                     # 返回 PyTorch 张量
    )
    
    # 将输入数据移动到 GPU 设备上进行推理
    inputs = inputs.to("cuda")

    # 使用模型生成回答文本
    # 禁用梯度计算以节省内存和提高速度（推理时不需要梯度）
    with torch.no_grad():
        try:
            # 调用模型的 generate 方法生成文本
            generated_ids = model.generate(
                **inputs,                               # 解包输入数据
                max_new_tokens=128,                     # 最大生成token数量
                do_sample=False,                        # 使用贪心解码，避免随机采样的数值问题
                temperature=1.0,                        # 温度参数（贪心解码时不起作用）
                top_p=1.0,                              # top-p 采样参数（贪心解码时不起作用）
                pad_token_id=tokenizer.eos_token_id,    # 填充token ID
                eos_token_id=tokenizer.eos_token_id,    # 结束token ID
                repetition_penalty=1.0                  # 重复惩罚系数
            )
        except Exception as e:
            # 处理生成过程中的异常（如CUDA内存不足、数值溢出等）
            print(f"生成失败: {e}")
            return "生成失败"
    
    # 提取新生成的token（去除输入部分）
    # generated_ids 包含输入+输出，
    # 切片out_ids[len(in_ids) :]去除输入序列 (in_ids) 长度的前缀，只保留新生成的部分
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 将生成的token ID解码为文本
    output_text = processor.batch_decode(
        generated_ids_trimmed,                    # 新生成的token IDs
        skip_special_tokens=True,                 # 跳过特殊token（如[PAD], [EOS]等）
        clean_up_tokenization_spaces=False        # 不清理分词空格
    )
    
    # 返回生成的文本（取第一个，因为batch_size=1）
    return output_text[0]


# ================= 模型和处理器加载 ====================
# 从本地路径加载预训练的 Qwen2.5-VL-7B-Instruct 模型
# 该模型是多模态视觉语言模型，能够同时处理图像和文本输入

# 加载分词器 (Tokenizer)
# 分词器负责将文本转换为模型可以理解的 token IDs
tokenizer = AutoTokenizer.from_pretrained(
    "/home/swq/Code/Qwen/models/Qwen/Qwen2.5-VL-7B-Instruct/", 
    use_fast=False,                                              # 使用慢速分词器，更稳定
    trust_remote_code=True                                       # 信任远程代码，允许加载自定义模型代码
)

# 加载多模态处理器 (Processor)
# 处理器能够同时处理文本和图像，将它们转换为模型输入格式
processor = AutoProcessor.from_pretrained("/home/swq/Code/Qwen/models/Qwen/Qwen2.5-VL-7B-Instruct")

# 加载预训练模型
# Qwen2_5_VLForConditionalGeneration 是条件生成模型，根据给定条件生成文本
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/swq/Code/Qwen/models/Qwen/Qwen2.5-VL-7B-Instruct/", 
    device_map="auto",                  # 自动分配到可用GPU（受CUDA_VISIBLE_DEVICES限制）
    torch_dtype=torch.float32,          # 使用FP32精度，避免数值不稳定问题
    trust_remote_code=True,             # 信任远程代码
)

# 启用输入梯度计算
# 当使用梯度检查点技术时需要调用此方法，允许输入也参与梯度计算
model.enable_input_require_grads()

# ================= 数据集处理 =================
# 读取并预处理训练数据集
# 数据集格式：JSON文件，每条记录包含对话形式的图像描述任务

train_json_path = "data_vl.json"                   # 原始数据文件路径
with open(train_json_path, 'r') as f:
    data = json.load(f)                            # 加载JSON数据
    # 数据集划分：最后12条作为测试集，其余作为训练集
    train_data = data[:-12]                        # 训练数据（除最后12条外的所有数据）
    test_data = data[-12:]                         # 测试数据（最后12条）

# 将划分后的数据保存为独立文件，便于后续加载和管理
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)                                     # 保存训练集

with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)                                      # 保存测试集

# 使用 HuggingFace Dataset 加载训练数据
# Dataset 提供了高效的数据加载和处理功能
train_ds = Dataset.from_json("data_vl_train.json")

# 对训练数据集应用预处理函数
# map() 会对数据集中的每一条记录调用 process_func 进行预处理
# 预处理包括：图像加载、文本分词、序列拼接、标签设置等
train_dataset = train_ds.map(process_func)

# ============== LoRA 微调配置 ==================
# LoRA (Low-Rank Adaptation) 是一种参数高效的微调技术
# 通过在预训练模型的线性层中插入低秩矩阵，只训练少量新增参数就能获得良好效果
# 相比全参数微调，LoRA 显著减少了训练参数量和显存需求

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,                # 任务类型：因果语言模型（自回归生成）
    target_modules=[                             # 目标模块：指定要应用LoRA的模型层
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力机制的查询、键、值、输出投影层
        "gate_proj", "up_proj", "down_proj"      # 前馈网络的门控、上升、下降投影层
    ],
    inference_mode=False,                        # 训练模式（非推理模式）
    r=64,                                        # LoRA秩：低秩矩阵的维度，影响参数量和表达能力
    lora_alpha=16,                               # LoRA缩放参数：控制LoRA输出的缩放强度
    lora_dropout=0.05,                           # LoRA层的Dropout比例，防止过拟合
    bias="none",                                 # 偏置参数处理方式：不训练偏置参数
)

# 将LoRA配置应用到原始模型，创建PEFT模型
# get_peft_model 会在指定的模块中插入LoRA层，冻结原始参数
peft_model = get_peft_model(model, config)

# ================================ 训练参数配置 ================================
# TrainingArguments 定义了模型训练的各种超参数和配置选项
# 这些参数对训练效果和稳定性有重要影响

args = TrainingArguments(
    output_dir="./output/Qwen2.5-VL-7B",         # 模型检查点和日志的保存目录
    
    # 批次大小和梯度累积设置
    per_device_train_batch_size=2,               # 每个GPU上的训练批次大小（减小以节省显存）
    gradient_accumulation_steps=8,               # 梯度累积步数（实际批次大小 = 2*8 = 16）
    
    # 日志记录设置
    logging_steps=10,                            # 每10步记录一次训练日志
    logging_first_step=5,                        # 第5步开始记录日志
    
    # 训练轮次和保存设置
    num_train_epochs=2,                          # 训练轮数：遍历数据集2次
    save_steps=100,                              # 每100步保存一次检查点
    save_on_each_node=True,                      # 在每个计算节点上保存检查点
    
    # 学习率和优化器设置
    learning_rate=5e-5,                          # 学习率：较小的值提升训练稳定性
    warmup_steps=10,                             # 预热步数：前10步逐渐增加学习率
    max_grad_norm=1.0,                           # 梯度裁剪：限制梯度最大范数，防止梯度爆炸
    
    # 内存和精度优化
    gradient_checkpointing=True,                 # 梯度检查点：用时间换显存，减少内存占用
    fp16=False,                                  # 禁用半精度浮点，避免数值不稳定
    dataloader_pin_memory=False,                 # 禁用内存锁定，避免CUDA内存问题
    
    # 监控和报告设置
    report_to="none",                            # 不向外部服务报告（使用SwanLab代替）
)
        
# ================================ 实验监控配置 ================================
# SwanLab 是一个机器学习实验跟踪平台，用于监控训练过程和结果
# 提供损失曲线、学习率变化、梯度分析等可视化功能

swanlab_callback = SwanLabCallback(
    project="Qwen2.5-VL-finetune",                  # 项目名称：在SwanLab平台上的项目标识
    experiment_name="qwen2.5-VL-coco2014",          # 实验名称：当前实验的标识
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct",  # 基础模型链接
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",  # 数据集链接
        "github": "https://github.com/datawhalechina/self-llm",  # 代码仓库链接
        "prompt": "COCO Yes: ",                                  # 使用的提示模板
        "train_data_number": len(train_data),                    # 训练数据量
        "lora_rank": 64,                                         # LoRA秩参数
        "lora_alpha": 16,                                        # LoRA缩放参数
        "lora_dropout": 0.05,                                     # LoRA dropout率
    },
)

# ================================ 训练器配置 ================================
# Trainer 是 HuggingFace Transformers 的训练器，集成了训练循环、验证、保存等功能
# 提供了统一的接口来管理整个训练过程

trainer = Trainer(
    model=peft_model,                               # 要训练的模型（应用了LoRA的PEFT模型）
    args=args,                                      # 训练参数配置
    train_dataset=train_dataset,                    # 训练数据集
    
    # 数据整理器：处理批次数据的填充和对齐
    data_collator=DataCollatorForSeq2Seq(           
        tokenizer=tokenizer,                        # 使用的分词器
        padding=True                                # 启用填充，将批次中的序列填充到相同长度
    ),
    callbacks=[swanlab_callback],                   # 回调函数列表：包含SwanLab监控
)

# ================================ 开始训练 ================================
# 启动模型训练过程
# 训练过程将按照配置的参数进行，包括前向传播、损失计算、反向传播、参数更新等
print("开始LoRA微调训练...")
trainer.train()

# ================================ 模型测试和验证 ================================
# 训练完成后，加载最新的检查点进行推理测试
# 验证模型在测试集上的表现，并将结果上传到SwanLab进行可视化

# 配置推理用的LoRA参数（与训练时相同，但设为推理模式）
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,                                # 任务类型：因果语言模型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
    inference_mode=True,                                         # 推理模式：冻结LoRA参数，不进行更新
    r=64,                                                        # LoRA秩（与训练时相同）
    lora_alpha=16,                                               # LoRA缩放参数（与训练时相同）
    lora_dropout=0.05,                                           # LoRA dropout率（推理时通常不起作用）
    bias="none",                                                 # 偏置处理方式
)

# 自动查找并加载最新的检查点
import glob                                                      # 用于文件路径匹配
checkpoint_dirs = glob.glob("./output/Qwen2.5-VL-7B/checkpoint-*")  # 查找所有检查点目录

if checkpoint_dirs:
    # 按检查点编号排序，选择最新的（编号最大的）
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    print(f"使用最新checkpoint: {latest_checkpoint}")
    
    # 从检查点加载训练好的LoRA模型
    val_peft_model = PeftModel.from_pretrained(
        model,                                                   # 基础模型
        model_id=latest_checkpoint,                              # 检查点路径
        config=val_config                                        # 推理配置
    )
else:
    print("未找到checkpoint，跳过测试")
    exit()

# =================== 测试数据推理 ====================
# 在测试集上进行推理，验证模型性能

# 加载测试数据集
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []                                             # 存储测试结果的列表
print("开始在测试集上进行推理...")

# 遍历测试集中的每个样本
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]       # 获取输入内容
    
    # 提取图像路径：去掉标记符号，只保留实际路径
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    # 构建推理用的消息格式（与训练时格式一致）
    messages = [{
        "role": "user",                                          # 用户角色
        "content": [                                             # 消息内容
            {
                "type": "image",                                 # 图像类型
                "image": origin_image_path                       # 图像路径
            },
            {
                "type": "text",                                  # 文本类型
                "text": "COCO Yes:"                              # 提示文本
            }
        ]
    }]
    
    # 使用训练好的模型进行推理
    response = predict(messages, val_peft_model)
    
    # 将模型回答添加到对话中
    messages.append({"role": "assistant", "content": f"{response}"})
    
    print(messages[-1])                       # 打印模型最后一条回答
    
    # 为SwanLab创建图像记录，包含图像和生成的描述
    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

# ================== 结果记录和保存 ================
# 将测试结果上传到SwanLab平台进行可视化展示

swanlab.log({"Prediction": test_image_list})                    # 上传预测结果图像

# 完成实验记录，释放资源
print("测试完成，正在保存实验记录...")
swanlab.finish()
