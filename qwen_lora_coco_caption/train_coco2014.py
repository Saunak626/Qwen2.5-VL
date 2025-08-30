"""
在 COCO 数据集上微调 Qwen2.5-VL-7B 模型

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

# =============== GPU 设备配置 ============
# 指定使用的GPU，此设置必须在导入torch前完成
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import torch
from datasets import Dataset
from modelscope import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import glob

# ================================ 数据预处理函数 ================================
def process_func(example):
    """
    对训练样本进行预处理，转换为模型输入格式。
    - 解析对话数据，提取图像和文本。
    - 构建多模态消息，并使用processor转换为token。
    - 拼接输入和输出，创建训练序列和标签。
    """

    MAX_LENGTH = 8192  # 最大序列长度限制
    
    input_ids, attention_mask, labels = [], [], [] # 初始化返回变量
    
    # 解析对话数据结构
    conversation = example["conversations"]  # 获取对话内容
    input_content = conversation[0]["value"]  # 用户输入（包含图像路径和文本）
    output_content = conversation[1]["value"]  # 模型应该生成的回答
    
    # 从输入内容中提取图像文件路径
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    # 构建多模态消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]
    
    # 使用 processor 将消息转换为聊天模板格式的文本
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # process_vision_info: Qwen-VL 专用函数，处理消息中的图像和视频
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 同时处理文本和视觉数据
    inputs = processor(
        text=[text],            # 文本数据（列表格式）
        images=image_inputs,    # 图像数据
        videos=video_inputs,    # 视频数据（本例中为空）
        padding=True,           # 填充到批次中最长序列的长度
        return_tensors="pt",    # 返回 PyTorch 张量格式
    )
    
    # 将 PyTorch 张量转换为列表以便拼接
    inputs = {key: value.tolist() for key, value in inputs.items()}
    
    # 分离指令部分和回答部分，为损失计算做准备
    instruction = inputs 
    
    response = tokenizer(f"{output_content}", add_special_tokens=False)

    # 构建完整的训练序列：[输入序列] + [输出序列] + [结束符]
    input_ids = (instruction["input_ids"][0] + response["input_ids"] + [tokenizer.eos_token_id])
    
    # 构建注意力掩码，标记哪些位置是有效的token（1表示有效，0表示填充）
    attention_mask = (
        instruction["attention_mask"][0] +     # 输入部分的注意力掩码
        response["attention_mask"] +           # 输出部分的注意力掩码
        [1]                                    # 结束符的注意力掩码
    )
    
    # 构建标签序列，用于计算损失
    # -100 是 PyTorch 交叉熵损失函数的忽略索引，表示不计算该位置的损失
    # 只有输出部分（模型应该生成的内容）才计算损失
    labels = (
            [-100] * len(instruction["input_ids"][0]) +    # 输入部分设为-100，不计算损失
            response["input_ids"] +                        # 输出部分计算损失
            [tokenizer.eos_token_id]                       # 结束符也要计算损失
    )
    
    # 截断防止超出模型最大长度限制
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]           # 截断输入序列
        attention_mask = attention_mask[:MAX_LENGTH] # 截断注意力掩码
        labels = labels[:MAX_LENGTH]                 # 截断标签序列

    # 转换为 PyTorch 张量，指定合适的数据类型
    input_ids = torch.tensor(input_ids, dtype=torch.long)       # token IDs（整数类型）
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)  # 注意力掩码（整数类型）
    
    labels = torch.tensor(labels, dtype=torch.long)             # 标签（整数类型）
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'], dtype=torch.float32)  # 图像像素值（浮点类型）
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 图像网格尺寸，去除批次维度  (1,h,w)->(h,w)
    
    # 返回处理后的训练样本，包含模型训练所需的所有输入
    return {
        "input_ids": input_ids,                          # 输入token序列
        "attention_mask": attention_mask,                # 注意力掩码
        "labels": labels,                                # 训练标签
        "pixel_values": inputs['pixel_values'],          # 图像像素数据
        "image_grid_thw": inputs['image_grid_thw']       # 图像网格信息
    }

# ================================ 推理预测函数 ================================
def predict(messages, model, processor, tokenizer):
    """
    使用训练好的模型进行推理预测。
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception as e:
            print(f"生成失败: {e}")
            return "生成失败"
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]


# ================= 模型和处理器加载 ====================
model_path = "/home/swq/Code/Qwen2.5-VL/models/Qwen/Qwen2.5-VL-7B-Instruct"

# 加载分词器 (Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False,
    trust_remote_code=True
)

# 加载多模态处理器 (Processor)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 加载预训练模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)


# ================= 数据集处理 =================
# 读取并预处理训练数据集
train_json_path = "data_vl.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    # 数据集划分：最后12条作为测试集，其余作为训练集
    train_data = data[:-12]
    test_data = data[-12:]

# 将划分后的数据保存为独立文件
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)

with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

# 使用 HuggingFace Dataset 加载训练数据
train_ds = Dataset.from_json("data_vl_train.json")

# 对训练数据集应用预处理函数
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# ============== LoRA 微调配置 ==================
# LoRA (Low-Rank Adaptation) 是一种参数高效的微调技术
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

# ================================ 关键修复：调整代码顺序 ================================
# 1. 首先，使用 get_peft_model 将基础模型包装成 PEFT 模型。
peft_model = get_peft_model(model, config)

# 2. 然后，对包装后的 peft_model 启用输入梯度计算。
#    这可以确保在使用梯度检查点时，梯度能够正确地在模型中传播。
#    `peft_model` 会将此调用正确地传递给其内部的基础模型。
peft_model.enable_input_require_grads()

# ================================ 训练参数配置 ================================
# TrainingArguments 定义了模型训练的各种超参数和配置选项
args = TrainingArguments(
    output_dir="./output/Qwen2.5-VL-7B",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    save_on_each_node=True,
    learning_rate=5e-5,
    warmup_steps=10,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    fp16=False,
    dataloader_pin_memory=False,
    report_to="none",
)
    
# ============================== SwanLab 实验监控配置 ==============================
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-VL-finetune",
    experiment_name="qwen2.5-VL-coco2014",
    config={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "dataset": "coco_2014_caption",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

# ================================ Trainer 配置 ================================
# Trainer 是 HuggingFace Transformers 的训练器，集成了训练循环
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True
    ),
    callbacks=[swanlab_callback],
)

# ================================ 启动微调 ================================
print("开始LoRA微调训练...")
trainer.train()

# ================================ 模型测试和验证 ================================
# 自动查找并加载最新的检查点
checkpoint_dirs = glob.glob("./output/Qwen2.5-VL-7B/checkpoint-*")

if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    print(f"使用最新checkpoint: {latest_checkpoint}")
    
    # 重新加载基础模型用于推理，确保模型状态干净
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # 将训练好的LoRA权重加载到基础模型上
    val_peft_model = PeftModel.from_pretrained(base_model, latest_checkpoint)
    val_peft_model.eval() # 设置为评估模式
else:
    print("未找到checkpoint，跳过测试")
    val_peft_model = None

# =================== 测试数据推理 ====================
if val_peft_model:
    with open("data_vl_test.json", "r") as f:
        test_dataset = json.load(f)

    test_image_list = []
    print("开始在测试集上进行推理...")

    for item in test_dataset:
        input_image_prompt = item["conversations"][0]["value"]
        origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        
        # 构建推理用的消息格式
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": "COCO Yes:"}
            ]
        }]
        
        response = predict(messages, val_peft_model, processor, tokenizer)
        
        print(f"图片: {origin_image_path} -> 回答: {response}")
        
        # 为SwanLab创建图像记录
        test_image_list.append(swanlab.Image(origin_image_path, caption=response))

    # ================== 结果记录和保存 ================
    if test_image_list:
        swanlab.log({"Prediction": test_image_list})

# 完成实验记录
print("训练和测试完成，正在保存实验记录...")
swanlab.finish()

