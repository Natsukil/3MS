from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
from pytorch_msssim import ssim
from evaluations.metrics import *


def extract_region(img, quadrant, size):
    """
    根据象限从图像中提取区域。
    :param img: 输入图像 [height, width]
    :param quadrant: 象限编号 (1, 2, 3, 4)
    :param size: 每个区域的尺寸 (height, width)
    :return: 提取的区域
    """
    h, w = size
    if quadrant == 1:
        return img[:h, :w]  # 左上
    elif quadrant == 2:
        return img[:h, -w:]  # 右上
    elif quadrant == 3:
        return img[-h:, :w]  # 左下
    elif quadrant == 4:
        return img[-h:, -w:]  # 右下


def evaluate_model(target, ref, device='cuda', binary_masks=None):
    """
    计算模型在给定输入和目标之间的性能指标。
    :param binary_masks:
    :param target: 模型输出
    :param ref: 目标图像
    :param device: 设备类型
    :return: 
    """
    assert target.shape == ref.shape, "Output and target must have the same shape"
    # assert binary_masks is not None, "Binary masks must be provided"
    batch_size, channels, height, width = target.shape

    region_size = (height // 2, width // 2)
    avg_psnr = [0.0] * 4  # 四个区域的PSNR平均值
    avg_ssim = [0.0] * 4  # 四个区域的SSIM平均值

    if torch.cuda.is_available() and device == 'cuda':
        psnr_func = psnr_gpu
        ssim_func = ssim
    else:
        psnr_func = psnr_np
        ssim_func = ssim_np

    # 遍历所有batch和slice
    for i in range(batch_size):
        for quadrant in range(1, 5):
            # 提取对应象限的target和output区域
            target_region = extract_region(ref[i, 0, :, :], quadrant, region_size)
            output_region = extract_region(target[i, 0, :, :], quadrant, region_size)

            # 调整维度以适应ssim函数要求
            target_region = target_region.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            output_region = output_region.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # 计算PSNR和SSIM
            cur_psnr = psnr_func(target_region, output_region)
            cur_ssim = ssim_func(target_region, output_region)

            avg_psnr[quadrant - 1] += cur_psnr
            avg_ssim[quadrant - 1] += cur_ssim

    # 计算平均值
    avg_psnr = [x / batch_size for x in avg_psnr]
    avg_ssim = [x / batch_size for x in avg_ssim]

    return avg_psnr, avg_ssim
