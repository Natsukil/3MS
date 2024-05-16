import pytorch_msssim
import torch
import torch.nn as nn


class loss_functions:
    def __init__(self):
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM()
    def calculate_loss(self, y_hat, y,  binary_masks):
        # assert binary_masks is not None, "Binary masks must be provided"

        # 假设输入的尺寸为 [batch_size, 1, 2*w, 2*h]
        w, h = y.shape[2] // 2, y.shape[3] // 2
        total_loss = 0

        # 从y_hat和y中提取四个区域
        regions = {
            't1c': (slice(None), slice(None), slice(0, w), slice(0, h)),  # 左上
            't1n': (slice(None), slice(None), slice(0, w), slice(h, 2 * h)),  # 右上
            't2w': (slice(None), slice(None), slice(w, 2 * w), slice(0, h)),  # 左下
            't2f': (slice(None), slice(None), slice(w, 2 * w), slice(h, 2 * h)),  # 右下
        }

        # 计算每个区域的损失
        for idx, region in enumerate(['t1c', 't1n', 't2w', 't2f']):
            # if binary_masks[idx] == '1':
            region_slice = regions[region]
            mse = self.mse_loss(y_hat[region_slice], y[region_slice])
            # ssim = self.ssim_loss(y_hat[region_slice], y[region_slice])

            # 你可以根据需要调整损失的权重或合并策略
            # combined_loss = mse + (1 - ssim)
            # total_loss += combined_loss
            total_loss += mse
        return total_loss