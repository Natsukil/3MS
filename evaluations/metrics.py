import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def psnr_np(target, ref):
    """
    归一化图像的数据范围是-1到1。
    假设图像的像素值范围为 [-1, 1]。

    """
    # 确保输入为float类型，以便进行精确计算
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    # 计算差异
    diff = ref_data - target_data

    # 计算均方误差（MSE）
    mse = np.mean(np.square(diff))

    # 计算PSNR，最大像素值设置为2
    psnr_value = 20 * np.log10(2.0 / np.sqrt(mse))
    return psnr_value


def psnr_gpu(target, output, max_pixel=1.0):
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(output, target)  # 确保output和target已经在GPU上
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def ssim_np(target, ref):
    """
    计算两个单通道归一化图像之间的 SSIM。
    假设图像的像素值范围为 [-1, 1]。

    :param target: 第一幅图像，numpy数组格式,
    :param ref: 第二幅图像，numpy数组格式。
    :return: SSIM值。
    """
    # 计算 SSIM
    # 我们指定 data_range 参数为 2，因为归一化图像的范围是从-1到1，差值为2
    ssim_value = ssim(target, ref, data_range=2)
    return ssim_value


def ssim_gpu(target, output):
    return ssim
