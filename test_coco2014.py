import torch
import os
import json
import glob
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType

# ================================ GPU 设备配置 ==============================
# 指定使用的GPU，确保与训练环境一致
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ================================ 推理预测函数 ================================
def predict(messages, model, processor):
    """
    使用模型进行推理预测。
    
    Args:
        messages (list): 符合Qwen-VL格式的多模态消息列表。
        model: 加载了LoRA权重的PEFT模型。
        processor: 用于处理文本和图像的处理器。
        
    Returns:
        str: 模型生成的文本回答。
    """
    # 将消息转换为模型输入格式
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    
    # 执行推理，不计算梯度
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    # 解码生成的token，得到文本结果
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text[0]

# ================================ 模型加载 ================================
# 定义基础模型和LoRA检查点的路径
BASE_MODEL_PATH = "/home/swq/Code/Qwen2.5-VL/models/Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = "./output/Qwen2.5-VL-7B"

# LoRA 配置 (必须与训练时完全一致)
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 设置为推理模式
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

# 1. 加载基础模型
print(f"正在从 {BASE_MODEL_PATH} 加载基础模型...")
# 使用与训练时相同的FP32精度
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, 
    torch_dtype=torch.float32, 
    device_map="auto", 
    trust_remote_code=True
)

# 2. 自动查找并加载最新的LoRA检查点
print(f"正在从 {OUTPUT_DIR} 查找最新的checkpoint...")
checkpoint_dirs = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
if not checkpoint_dirs:
    print(f"错误: 在 {OUTPUT_DIR} 目录下未找到任何checkpoint。请确认训练已完成并生成了检查点。")
    exit()

# 根据编号找到最新的检查点
latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
print(f"加载最新的LoRA权重: {latest_checkpoint}")

# 3. 将LoRA权重合并到基础模型
model = PeftModel.from_pretrained(model, model_id=latest_checkpoint, config=config)

# 4. 加载处理器
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
print("模型和处理器加载完成。")


# ================================ 数据加载与推理 ================================
# 加载由训练脚本生成的测试数据集
TEST_DATA_PATH = "data_vl_test.json"
if not os.path.exists(TEST_DATA_PATH):
    print(f"错误: 测试文件 {TEST_DATA_PATH} 不存在。请先运行训练脚本生成测试文件。")
    exit()

with open(TEST_DATA_PATH, "r") as f:
    test_dataset = json.load(f)

print(f"\n开始在 {len(test_dataset)} 条测试数据上进行推理...")

# 遍历测试集中的每个样本进行推理
for i, item in enumerate(test_dataset):
    print(f"\n--- 正在处理样本 {i+1}/{len(test_dataset)} ---")
    
    # 提取图像路径和真实的描述
    input_image_prompt = item["conversations"][0]["value"]
    ground_truth = item["conversations"][1]["value"]
    
    origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    print(f"  图像: {origin_image_path}")
    print(f"  真实描述: {ground_truth}")

    # 构建推理所需的消息格式
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": origin_image_path},
            {"type": "text", "text": "COCO Yes:"}
        ]
    }]
    
    # 进行预测
    response = predict(messages, model, processor)
    print(f"  模型预测: {response}")
    
print("\n所有测试样本处理完毕。")
