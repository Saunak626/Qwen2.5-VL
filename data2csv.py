# 导入所需的库
from modelscope.msdatasets import MsDataset
import os
import pandas as pd

MAX_DATA_NUMBER = 15000  # 最大处理图片数量

# 从modelscope下载COCO 2014图像描述数据集
print("正在加载COCO 2014数据集...")
ds = MsDataset.load('modelscope/coco_2014_caption', split='train', download_mode="force_redownload")

# 显示数据集大小（修复：使用len(ds)而不是ds[0]）
dataset_size = len(ds)
print(f"数据集实际大小: {dataset_size} 条")

# 检查数据集结构
if dataset_size > 0:
    sample = ds[0]
    print(f"数据字段: {list(sample.keys())}")
    
    # 检查是否包含预期字段
    has_image_id = 'image_id' in sample
    has_caption = 'caption' in sample
    
    if not has_image_id or not has_caption:
        print("⚠️  警告: 数据集缺少预期字段")
        if not has_image_id:
            print("   缺少 'image_id' 字段")
        if not has_caption:
            print("   缺少 'caption' 字段")
        print("   将使用替代方案处理数据")

# 设置计划处理的图片数量上限
total_to_process = min(MAX_DATA_NUMBER, dataset_size)
print(f"计划处理数量: {total_to_process} 条")

# 检查目录是否已存在
if not os.path.exists('coco_2014_caption'):
    print('coco_2014_caption目录不存在，开始数据处理...')
    # 创建保存图片的目录
    os.makedirs('coco_2014_caption', exist_ok=True)

    # 初始化存储图片路径和描述的列表
    image_paths = []
    captions = []
    start_index = 0
    existing_count = 0 # 目录不存在，已有数据为0

else:
    print('coco_2014_caption目录已存在。')
    # 检查是否已存在CSV文件并读取
    csv_path = './coco-2024-dataset.csv'
    if os.path.exists(csv_path):
        print(f'正在读取已有的数据文件: {csv_path}')
        df_existing = pd.read_csv(csv_path)
        existing_count = len(df_existing) # 已有数据量
        image_paths = df_existing['image_path'].tolist()
        captions = df_existing['caption'].tolist()
        start_index = existing_count
        print(f'已存在 {existing_count} 条数据。')
    else:
        coco_dir = os.path.abspath("coco_2014_caption")
        print(f'目录 {coco_dir} 已存在但未找到 {csv_path} 文件，将重新处理。')
        existing_count = 0
        image_paths = []
        captions = []
        start_index = 0

# 计算还需要处理的数据量
num_to_add = total_to_process - existing_count

if num_to_add > 0:
    print(f'计划增量添加 {num_to_add} 条数据，从索引 {start_index} 开始处理...')
    for i in range(start_index, total_to_process):
        # 获取每个样本的信息
        item = ds[i]
        
        # 获取数据（兼容不同的数据集结构）
        if 'image_id' in item:
            image_id = item['image_id']
        else:
            # 如果没有image_id字段，使用索引作为ID
            image_id = f"image_{i:06d}"
        
        if 'caption' in item:
            caption = item['caption']
        else:
            # 如果没有caption字段，使用默认描述
            caption = f"Image {i+1} from COCO dataset"
        
        image = item['image']

        # 保存图片并记录路径
        image_path = os.path.abspath(f'coco_2014_caption/{image_id}.jpg')
        # 检查图片是否已存在，避免重复保存
        if not os.path.exists(image_path):
             image.save(image_path)

        # 将路径和描述添加到列表中
        image_paths.append(image_path)
        captions.append(caption)

        # 每处理50张图片打印一次进度
        if (i + 1) % 50 == 0:
            progress = (i + 1) / total_to_process * 100
            print(f'处理进度: {i+1}/{total_to_process} ({progress:.1f}%)')

    # 将更新后的图片路径和描述保存为CSV文件
    df_updated = pd.DataFrame({
        'image_path': image_paths,
        'caption': captions
    })

    # 将数据保存为CSV文件
    df_updated.to_csv('./coco-2024-dataset.csv', index=False)

    print(f'数据处理完成，共处理了 {len(df_updated)} 条数据 (含原有数据)。')
else:
     print('已处理的数据量达到或超过设定的最大处理数量，无需增量处理。')
     print(f'当前已处理数据量: {existing_count}, 最大处理数量: {MAX_DATA_NUMBER}')
     print(f'数据集实际大小: {dataset_size}')