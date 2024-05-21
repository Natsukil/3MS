import pytorch_msssim
import torch.nn as nn


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
    def __init__(self):
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM()

    def calculate_loss(self, y_hat, y,  binary_masks):
        assert binary_masks is not None, "Binary masks must be provided"

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

        # 计算每个区域的权重
        weights = calculate_weights(binary_masks)

        # 计算每个区域的损失
        for idx, region in enumerate(['t1c', 't1n', 't2w', 't2f']):
            # if binary_masks[idx] == '1':
            region_slice = regions[region]
            mse = self.mse_loss(y_hat[region_slice], y[region_slice])
            # ssim = self.ssim_loss(y_hat[region_slice], y[region_slice])

            # 你可以根据需要调整损失的权重或合并策略
            # combined_loss = mse + (1 - ssim)
            # total_loss += combined_loss
            total_loss += weights[idx] * mse

        return total_loss

# if __name__=='__main__':
#     loss = LossFunctions()
#     print(calculate_weights(binary_masks='1000'))
