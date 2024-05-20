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
from mask_generator.masker import random_masked_area


class Dataset_brats(Dataset):
    def __init__(self, root_dir, slice_deep, slice_size=192, mask_kernel_size=12, binary_mask='1111', mask_rate=0.5,
                 mode='train'):
        """
        初始化函数，列出所有患者的数据目录。
        """
        self.mode = mode
        self.root_dir = root_dir
        self.patients = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                         os.path.isdir(os.path.join(root_dir, name))]
        # 遮蔽算法的超参数初始化
        self.slice_size = slice_size
        self.mask_kernel_size = mask_kernel_size
        self.binary_mask = binary_mask
        self.mask_rate = mask_rate
        self.slice_deep = slice_deep

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        """
        根据索引idx获取数据，处理后返回。
        """
        patient_path = self.patients[idx]
        # 预处理并拼接图像
        combined_image = self.preprocess_directory(patient_path)
        # 生成遮蔽掩码
        masked_image = random_masked_area(combined_image, self.mask_kernel_size, self.slice_size, self.binary_mask,
                                          self.mask_rate)
        # 生成遮蔽后图像
        masked_result = np.where(masked_image == 1, combined_image, -1)
        # 返回遮蔽后图像X和原始图像y
        # Convert numpy array to torch tensor
        return torch.tensor(masked_result, dtype=torch.float32), torch.tensor(combined_image, dtype=torch.float32)

    def preprocess_directory(self, directory):
        """
        处理指定目录下的所有图像，返回预处理和拼接后的图像。
        """
        # 读取图像
        # seg = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-seg.nii.gz"))
        t1c = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t1c.nii.gz"))
        t1n = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t1n.nii.gz"))
        t2w = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t2w.nii.gz"))
        t2f = self.read_image(os.path.join(directory, f"{os.path.basename(directory)}-t2f.nii.gz"))

        # 归一化、剪裁、随机翻转和拼接
        t1c = self.resize_and_crop(self.normalize(t1c), slice_deep=self.slice_deep, slice_size=self.slice_size)
        t1n = self.resize_and_crop(self.normalize(t1n), slice_deep=self.slice_deep, slice_size=self.slice_size)
        t2w = self.resize_and_crop(self.normalize(t2w), slice_deep=self.slice_deep, slice_size=self.slice_size)
        t2f = self.resize_and_crop(self.normalize(t2f), slice_deep=self.slice_deep, slice_size=self.slice_size)

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

    def resize_and_crop(self, image, slice_deep=128, slice_size=192):
        """
        从图像中均匀采样多个切片，并返回包含这些切片的新图像。

        :param image: 输入的 3D 图像，形状为 (depth, height, width)
        :param slice_deep: 要裁剪的深度大小
        :param slice_size: 要裁剪的高度、宽度大小
        :return: 新的 3D 图像，包含从原始图像中均匀采样的切片
        """

        # 计算图像的中间位置
        center_d = image.shape[0] // 2
        start_d = max(center_d - slice_deep // 2, 0)

        # 中间裁剪
        cropped_center = image[start_d:start_d + slice_deep, :, :]

        # 确保裁剪的高度和宽度在范围内
        start_size = max((image.shape[1] - slice_size) // 2, 0)
        cropped_center = cropped_center[:, start_size:start_size + slice_size, start_size:start_size + slice_size]

        return cropped_center

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


def get_brats_dataloader(root_dir, batch_size=1, slice_deep=16.
                         , slice_size=192, num_workers=1, mask_kernel_size=12,
                         binary_mask='1111', mask_rate=0.5, mode='train'):
    is_shuffle = False
    if mode == 'train':
        is_shuffle = True
    dataset = Dataset_brats(root_dir=root_dir, slice_deep=slice_deep, slice_size=slice_size,
                            binary_mask=binary_mask, mask_kernel_size=mask_kernel_size, mask_rate=mask_rate, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader


