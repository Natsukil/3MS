import os
import json
import random
from pathlib import Path

# 设置随机种子以保证可重复性
random.seed(42)

# 定义数据集路径
train_data_path = Path('data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
val_data_path = Path('data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData')

# 获取所有样本的文件夹名称
train_samples = [f.name for f in train_data_path.iterdir() if f.is_dir()]
test_samples = [f.name for f in val_data_path.iterdir() if f.is_dir()]

# 按照主序和次序进行升序排列
def sort_key(sample):
    parts = sample.split('-')
    main_seq = int(parts[2])
    sub_seq = int(parts[3])
    return (main_seq, sub_seq)

train_samples.sort(key=sort_key)
test_samples.sort(key=sort_key)

# 设置使用比例
usage_ratio = 0.5
train_samples = random.sample(train_samples, int(len(train_samples) * usage_ratio))
test_samples = random.sample(test_samples, int(len(test_samples) * usage_ratio))

# 划分数据集比例
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# 计算训练集和验证集的大小
total_train_samples = len(train_samples)
train_size = int(total_train_samples * train_ratio)
val_size = int(total_train_samples * val_ratio)

# 划分训练集和验证集
train_subset = train_samples[:train_size]
val_subset = train_samples[train_size:train_size + val_size]
test_subset = test_samples[:len(test_samples)]

# 创建所有样本的字典，仅包含子文件夹的文件名
data_splits = {
    'train': train_subset,
    'val': val_subset,
    'test': test_subset
}

# 保存为JSON文件
output_file = 'data_splits_pre_experiment.json'
with open(output_file, 'w') as f:
    json.dump(data_splits, f, indent=4)

print(f"数据集划分并保存到 {output_file} 文件中。")

# 将训练集、验证集和测试集分别保存为三个文件
def save_to_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

save_to_file('train.csv', data_splits['train'])
save_to_file('valid.csv', data_splits['val'])
save_to_file('test.csv', data_splits['test'])

print(f"训练集数量: {len(data_splits['train'])}")
print(f"验证集数量: {len(data_splits['val'])}")
print(f"测试集数量: {len(data_splits['test'])}")

# 输出验证集第20个和测试集第10个文件夹名
if len(data_splits['val']) >= 20:
    print(f"验证集第20个文件夹名: {data_splits['val'][19]}")
else:
    print("验证集不足20个样本。")

if len(data_splits['test']) >= 10:
    print(f"测试集第10个文件夹名: {data_splits['test'][9]}")
else:
    print("测试集不足10个样本。")
