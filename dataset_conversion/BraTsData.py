"""
数据集处理
文件路径下的每一个子文件夹代表一个人的数据，每个子文件夹下有5个文件，分别代表分割掩码、t1c序列、t1n序列、t2f序列、t2w序列，每个文件的格式为.nii.gz
文件结构如下
ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
|--BraTS-GLI-00000-000
|     |--BraTS-GLI-00000-000-seg.nii.gz
|     |--BraTS-GLI-00000-000-t1c.nii.gz
|     |--BraTS-GLI-00000-000-t1n.nii.gz
|     |--BraTS-GLI-00000-000-t2f.nii.gz
|     |--BraTS-GLI-00000-000-t2w.nii.gz
|--BraTS-GLI-00002-000
...
每个样本4个模态（序列）,每个序列155个切片,即每个文件为155张240*240的序列图像
需要对每个序列进行归一化和缩小至128张图像，每张图像以中心为准，剪裁至192*192
然后进行数据增强，在高度、宽度方向进行随机翻转、每个序列做同样的处理
最后将四个序列在128这个维度中相同维度的图像进行拼接，称为128*480*480的图像，顺序为左上t1c、右上t1n、左下t2w、右下t2f

1. 保留包含脑组织的切片，去除前后22与23张切片
2. 应用MRI强度标准化（零均值和单位方差）
3. 通过三次插值将像素调整为224*224

"""
import os
import numpy as np
import torch
import random
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader

class Dataset_brats(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        初始化函数，列出所有患者的数据目录。
        """
        self.mode = mode
        self.root_dir = root_dir
        self.patients = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                         os.path.isdir(os.path.join(root_dir, name))]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        """
        根据索引idx获取数据，处理后返回。
        """
        patient_path = self.patients[idx]
        combined_image = self.preprocess_directory(patient_path)
        return torch.tensor(combined_image, dtype=torch.float32)  # Convert numpy array to torch tensor

    def preprocess_directory(self, directory):
        """
        处理指定目录下的所有图像，返回预处理和拼接后的图像。
        """
        # 读取图像
        seg = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-seg.nii.gz"))
        t1c = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t1c.nii.gz"))
        t1n = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t1n.nii.gz"))
        t2w = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t2w.nii.gz"))
        t2f = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t2f.nii.gz"))

        # 归一化、剪裁、随机翻转和拼接
        t1c = self.resize_and_crop(self.normalize(t1c))
        t1n = self.resize_and_crop(self.normalize(t1n))
        t2w = self.resize_and_crop(self.normalize(t2w))
        t2f = self.resize_and_crop(self.normalize(t2f))

        # 决定一个随机翻转操作并应用到所有图像
        flip_action = random.choice([0, 1, 2])  # 从三种操作中随机选择
        t1c = self.random_flip(t1c, flip_action)
        t1n = self.random_flip(t1n, flip_action)
        t2w = self.random_flip(t2w, flip_action)
        t2f = self.random_flip(t2f, flip_action)

        # 合并图像
        top_row = np.concatenate((t1c, t1n), axis=2)  # 横向拼接
        bottom_row = np.concatenate((t2w, t2f), axis=2)
        combined_image = np.concatenate((top_row, bottom_row), axis=1)  # 纵向拼接
        return combined_image

    def read_image(self, path):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def normalize(self, image):
        max_val = np.percentile(image, 99)
        image = np.clip(image, 0, max_val)
        image = (image / max_val) * 2 - 1
        return image

    def resize_and_crop(self, image, new_depth=128, new_height=192, new_width=192):
        start_d = (image.shape[0] - new_depth) // 2
        start_h = (image.shape[1] - new_height) // 2
        start_w = (image.shape[2] - new_width) // 2
        return image[start_d:start_d + new_depth, start_h:start_h + new_height, start_w:start_w + new_width]

    def random_flip(self, image, action):
        """
        根据action参数翻转图像。
        action = 0 -> 不变
        action = 1 -> 左右翻转
        action = 2 -> 上下翻转
        """
        if action == 1:
            image = image[:, :, ::-1]  # 左右翻转
        elif action == 2:
            image = image[:, ::-1, :]  # 上下翻转
        return image


# def get_dataloader(root_dir, batch_size=1, shuffle=True, num_workers=1):
#     dataset = Dataset_brats(root_dir=root_dir)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
#     return dataloader

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     root_dir = 'E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
#     dataloader = get_dataloader(root_dir, batch_size=4)
#     data_iter = iter(dataloader)
#     images = next(data_iter)  # 获取一个批次的图像
#     image_to_show = images[0][0]  # 这里的0代表批次中的第一张图像，再一个0代表128个切片中的第一个
#
#     # 显示图像
#     plt.imshow(image_to_show, cmap='gray')  # 使用灰度颜色映射
#     plt.title('Sample Image from DataLoader')
#     plt.axis('off')  # 关闭坐标轴显示
#     plt.show()
