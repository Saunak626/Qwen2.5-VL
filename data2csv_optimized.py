import os
import pandas as pd
from modelscope.msdatasets import MsDataset

class CocoDatasetProcessor:
    """COCO数据集处理器"""
    
    def __init__(self, max_data_number=15000, output_dir='coco_2014_caption', csv_filename='coco-2024-dataset.csv'):
        self.max_data_number = max_data_number
        self.output_dir = output_dir
        self.csv_filename = csv_filename
        self.csv_path = f'./{csv_filename}'
        
    def load_dataset(self):
        """加载数据集并检查其结构"""
        print("正在加载COCO 2014数据集...")
        
        # 加载数据集
        self.ds = MsDataset.load('modelscope/coco_2014_caption', split='train')
        
        # 显示正确的数据集大小
        dataset_size = len(self.ds)
        print(f"数据集实际大小: {dataset_size} 条")
        
        # 检查数据结构
        if dataset_size > 0:
            sample = self.ds[0]
            print(f"数据字段: {list(sample.keys())}")
            
            # 检查是否包含预期字段
            if 'image_id' not in sample and 'caption' not in sample:
                print("⚠️  警告: 该数据集只包含图片，没有image_id和caption字段")
                print("   这不是完整的COCO标注数据集，无法提取描述信息")
                print("   建议使用包含完整标注的COCO数据集")
                return False
        
        # 设置实际处理数量
        self.total_to_process = min(self.max_data_number, dataset_size)
        print(f"计划处理数量: {self.total_to_process} 条")
        
        return True
    
    def check_existing_data(self):
        """检查已存在的数据"""
        self.existing_count = 0
        self.image_paths = []
        self.captions = []
        self.start_index = 0
        
        # 检查输出目录
        if not os.path.exists(self.output_dir):
            print(f'{self.output_dir}目录不存在，将创建新目录')
            os.makedirs(self.output_dir, exist_ok=True)
            return
            
        print(f'{self.output_dir}目录已存在')
        
        # 检查CSV文件
        if os.path.exists(self.csv_path):
            print(f'正在读取已有数据文件: {self.csv_path}')
            df_existing = pd.read_csv(self.csv_path)
            self.existing_count = len(df_existing)
            self.image_paths = df_existing['image_path'].tolist()
            self.captions = df_existing['caption'].tolist()
            self.start_index = self.existing_count
            print(f'已存在 {self.existing_count} 条数据')
        else:
            print(f'未找到 {self.csv_path} 文件，将重新处理')
    
    def process_dataset(self):
        """处理数据集"""
        if not self.load_dataset():
            return False
            
        self.check_existing_data()
        
        # 计算需要新增的数据量
        num_to_add = self.total_to_process - self.existing_count
        
        if num_to_add <= 0:
            print('已处理的数据量达到或超过设定的最大处理数量，无需增量处理')
            print(f'当前已处理数据量: {self.existing_count}, 最大处理数量: {self.max_data_number}')
            print(f'数据集实际大小: {len(self.ds)}')
            return True
        
        print(f'计划增量添加 {num_to_add} 条数据，从索引 {self.start_index} 开始处理...')
        
        # 处理新数据
        for i in range(self.start_index, self.total_to_process):
            item = self.ds[i]
            
            # 获取数据 - 针对当前数据集结构调整
            if 'image_id' in item:
                image_id = item['image_id']
            else:
                # 如果没有image_id，使用索引作为ID
                image_id = f"image_{i:06d}"
            
            if 'caption' in item:
                caption = item['caption']
            else:
                # 如果没有caption，使用默认描述
                caption = f"Image {i+1} from COCO dataset"
            
            image = item['image']
            
            # 保存图片
            image_path = os.path.abspath(f'{self.output_dir}/{image_id}.jpg')
            if not os.path.exists(image_path):
                image.save(image_path)
            
            # 记录数据
            self.image_paths.append(image_path)
            self.captions.append(caption)
            
            # 显示进度
            if (i + 1) % 50 == 0:
                progress = (i + 1) / self.total_to_process * 100
                print(f'处理进度: {i+1}/{self.total_to_process} ({progress:.1f}%)')
        
        # 保存结果
        self.save_results()
        return True
    
    def save_results(self):
        """保存处理结果"""
        df_updated = pd.DataFrame({
            'image_path': self.image_paths,
            'caption': self.captions
        })
        
        df_updated.to_csv(self.csv_path, index=False)
        print(f'数据处理完成，共处理了 {len(df_updated)} 条数据')
        print(f'结果已保存到: {self.csv_path}')

def main():
    """主函数"""
    processor = CocoDatasetProcessor(max_data_number=15000)
    
    try:
        success = processor.process_dataset()
        if not success:
            print("\n建议解决方案:")
            print("1. 寻找包含完整标注的COCO 2014数据集")
            print("2. 使用其他包含image_id和caption的数据集")
            print("3. 或者修改代码以适应当前数据集结构")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        print("请检查数据集是否正确加载")

if __name__ == "__main__":
    main() 