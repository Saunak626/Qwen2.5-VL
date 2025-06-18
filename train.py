import torch
import os

from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration, # 
    AutoProcessor,
)
import swanlab
import json

# 设置使用特定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用GPU 3

# 显示GPU信息
print(f"使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  总显存: {total_memory:.1f} GB")

def process_func(example):
    """
    对每一条数据进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 截取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": "COCO Yes:"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.eos_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.eos_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'], dtype=torch.float32)
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
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
    inputs = inputs.to("cuda")

    # 生成输出
    with torch.no_grad():
        try:
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False,  # 改为贪心解码避免采样问题
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0
            )
        except Exception as e:
            print(f"生成失败: {e}")
            return "生成失败"
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# 在modelscope上下载Qwen25-VL模型到本地目录下

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("/home/swq/Code/Qwen/models/Qwen/Qwen2.5-VL-7B-Instruct/", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/home/swq/Code/Qwen/models/Qwen/Qwen2.5-VL-7B-Instruct")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/home/swq/Code/Qwen/models/Qwen/Qwen2.5-VL-7B-Instruct/", device_map={"": 0}, torch_dtype=torch.float32, trust_remote_code=True,)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
# 拆分成训练集和测试集
train_json_path = "data_vl.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-12]
    test_data = data[-12:]

# 保存为data_vl_train.json和data_vl_test.json
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)

with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json("data_vl_train.json")
train_dataset = train_ds.map(process_func)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen2.5-VL-7B",
    per_device_train_batch_size=2,  # 减小batch size避免内存问题
    gradient_accumulation_steps=8,  # 增加梯度累积步数保持总batch size
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=5e-5,  # 降低学习率提升训练稳定性
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    max_grad_norm=1.0,  # 添加梯度裁剪
    warmup_steps=10,  # 添加warmup步数
    fp16=False,  # 禁用fp16避免数值不稳定
    dataloader_pin_memory=False,  # 禁用pin_memory避免CUDA内存问题
)
        
# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-VL-finetune",
    experiment_name="qwen2.5-VL-coco2014",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "COCO Yes: ",
        "train_data_number": len(train_data),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()

# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型
import glob
checkpoint_dirs = glob.glob("./output/Qwen2.5-VL-7B/checkpoint-*")
if checkpoint_dirs:
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    print(f"使用最新checkpoint: {latest_checkpoint}")
    val_peft_model = PeftModel.from_pretrained(model, model_id=latest_checkpoint, config=val_config)
else:
    print("未找到checkpoint，跳过测试")
    exit()

# 读取测试数据
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    messages = [{
        "role": "user", 
        "content": [
            {
            "type": "image", 
            "image": origin_image_path
            },
            {
            "type": "text",
            "text": "COCO Yes:"
            }
        ]}]
    
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])

    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

swanlab.log({"Prediction": test_image_list})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
