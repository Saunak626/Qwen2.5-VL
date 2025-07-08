from datasets import load_dataset

# 加载数据集
ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo", cache_dir="cache", token=True)

print(ds['train'][:1]) 
