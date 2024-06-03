import pytorch_msssim
import torch.nn as nn
import torch

def calculate_weights(binary_masks):
    """
    计算每个区域的权重
    :param binary_masks: 四位二进制字符串，表示哪些区域被遮蔽
    :return: 每个区域的权重列表
    """
    mask_counts = sum(int(b) for b in binary_masks)
    if mask_counts == 4:  # 所有区域均发生遮蔽
        weights = [0.25] * 4  # 均匀分配权重
    else:
        # 遮蔽区域和非遮蔽区域的比率为3:1
        # 每个遮蔽区域的权重
        masked_weight = 0.8 / mask_counts
        # 每个非遮蔽区域的权重
        unmasked_weight = 0.2 / (4 - mask_counts)
        weights = [masked_weight if b == '1' else unmasked_weight for b in binary_masks]
    return weights



class LossFunctions:
    def __init__(self, concat_method):
        self.background_weight = 0.01
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ssim_loss = pytorch_msssim.SSIM()
        self.concat = concat_method

    def calculate_loss_regions(self, y_hat, y, binary_masks):
        assert binary_masks is not None, "Binary masks must be provided"

        background_masks = (y > 0).float()
        total_loss = 0
        if self.concat == "plane":
            # 假设输入的尺寸为 [batch_size, 1, 2*w, 2*h]
            w, h = y.shape[2] // 2, y.shape[3] // 2
            # 从y_hat和y中提取四个区域
            regions = {
                't1c': (slice(None), slice(None), slice(0, w), slice(0, h)),  # 左上
                't1n': (slice(None), slice(None), slice(0, w), slice(h, 2 * h)),  # 右上
                't2w': (slice(None), slice(None), slice(w, 2 * w), slice(0, h)),  # 左下
                't2f': (slice(None), slice(None), slice(w, 2 * w), slice(h, 2 * h)),  # 右下
            }
        elif self.concat == "channels":
            regions = {
                't1c': (slice(None), slice(0, 1), slice(None), slice(None)),  # 第1个通道
                't1n': (slice(None), slice(1, 2), slice(None), slice(None)),  # 第2个通道
                't2w': (slice(None), slice(2, 3), slice(None), slice(None)),  # 第3个通道
                't2f': (slice(None), slice(3, 4), slice(None), slice(None)),  # 第4个通道
            }
        else:
            raise ValueError(f"Invalid concat mode: {self.concat}")

        # 计算每个区域的权重
        weights = calculate_weights(binary_masks)

        # 计算每个区域的损失
        for idx, region in enumerate(['t1c', 't1n', 't2w', 't2f']):
            # if binary_masks[idx] == '1':
            region_slice = regions[region]
            region_background_mask = background_masks[region_slice]

            mse = self.mse_loss(y_hat[region_slice], y[region_slice])
            # ssim = self.ssim_loss(y_hat[region_slice], y[region_slice])

            # 计算非背景部分的 MSE 损失
            non_background_mse = mse * region_background_mask
            non_background_loss = torch.sum(non_background_mse) / (torch.sum(region_background_mask) + 1e-8)

            # 计算背景部分的 MSE 损失
            background_mse = mse * (1 - region_background_mask)
            background_loss = torch.sum(background_mse) / (torch.sum(1 - region_background_mask) + 1e-8)

            # 加权非背景和背景损失
            combined_loss = (1 - self.background_weight) * non_background_loss + self.background_weight * background_loss

            # 根据权重合并每个区域的损失
            # 你可以根据需要调整损失的权重或合并策略
            # combined_loss = mse + (1 - ssim)
            # total_loss += combined_loss
            total_loss += weights[idx] * non_background_loss
            # total_loss += weights[idx] * mse.mean()/

        return total_loss

    # def calculate_loss_no_background(self, y_hat, y):
    #     """
    #     计算没有背景的损失。
    #
    #     参数:
    #     - y_hat: 预测的图像张量，形状为 (N, C, H, W)
    #     - y: 原始图像张量，形状为 (N, C, H, W)
    #
    #     返回:
    #     - loss: 计算出的损失
    #     """
    #     # 确保 binary_masks 在相同的设备上
    #     device = y.device
    #     target_masks = (y > 0).float().to(device)
    #
    #     # 计算所有像素点的MSE
    #     mse = self.mse_loss(y_hat, y)
    #
    #     # 使用 binary_masks 来选择非背景部分的 MSE
    #     non_background_mse = mse * target_masks
    #
    #     # 计算非背景部分的 MSE 平均值
    #     # 为避免除以零的情况，添加一个非常小的值到分母
    #     loss = torch.sum(non_background_mse) / (torch.sum(target_masks) + 1e-8)
    #
    #     return loss


# if __name__=='__main__':
#     loss = LossFunctions()
#     y_hat = torch.randint(0, 2, (1, 1, 4, 4)).float()
#     y = torch.randint(0, 2, (1, 1, 4, 4)).float()
#     loss_ = loss.calculate_loss_regions(y_hat, y, binary_masks='1111')
#     print(calculate_weights(binary_masks='1000'))
