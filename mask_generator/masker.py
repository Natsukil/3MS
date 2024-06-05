import math
import random
import numpy as np
import time
import torch

# from datasets import BraTsData
def random_masked_area(image_batch, mask_kernel_size, slice_size, binary_mask, mask_rate):
    """
    为图像批次创建遮蔽区域。
    :param method:
    :param image_batch: ndarray, 图像的批次，形状为 (slice_num, height, width)
    :param mask_kernel_size: int, 遮蔽块的尺寸
    :param slice_size: tuple, 每个子图的尺寸 (height, width)
    :param binary_mask: str, 四位二进制字符串，表示哪些区域需要遮蔽
    :param mask_rate: float, 遮蔽的比例
    :return: ndarray, 遮蔽后的图像批次
    """

    slice_num = image_batch.shape[0]
    # 初始化原图的遮蔽掩码

    masked_image = np.zeros(image_batch.shape)

    # 构建遮蔽块计数图的维度
    masked_sub_image_size = slice_size // mask_kernel_size

    total_block_count = masked_sub_image_size ** 2

    # 检查每一子图是否需要遮蔽
    # for index in range(batch_size):
    for slice_index in range(slice_num):

        masked_sub_image_count = np.ones((masked_sub_image_size, masked_sub_image_size))

        for i in range(4):
            if binary_mask[i] == '0':
                continue
                # 计算子图的位置
            if method == 'plane':
                row_start = (i // 2) * slice_size
                col_start = (i % 2) * slice_size

            # 收集所有可以被选择为保留块的位置
            all_blocks = [(r, c) for r in range(0, slice_size, mask_kernel_size)
                          for c in range(0, slice_size, mask_kernel_size)
                          if masked_sub_image_count[r // mask_kernel_size, c // mask_kernel_size] > 0]

            random.shuffle(all_blocks)  # 随机化块的顺序

            # 根据rate计算需要保留的块数
            keep_count = min(int((1 - mask_rate) * total_block_count), len(all_blocks))

            # 选择块并设置保留
            selected_blocks = all_blocks[:keep_count]
            if method == 'channels':
                for row, col in selected_blocks:
                    masked_image[slice_index, i, row:row + mask_kernel_size,
                                 col:col + mask_kernel_size] = 1  # 设置保留块
                    masked_sub_image_count[row // mask_kernel_size:(row // mask_kernel_size) + 1,
                                           col // mask_kernel_size:(col // mask_kernel_size) + 1] -= 1
            elif method == 'plane':
                for row, col in selected_blocks:
                    masked_image[slice_index, row_start + row:row_start + row + mask_kernel_size,
                                 col_start + col:col_start + col + mask_kernel_size] = 1  # 设置保留块
                    masked_sub_image_count[row // mask_kernel_size:(row // mask_kernel_size) + 1,
                                           col // mask_kernel_size:(col // mask_kernel_size) + 1] -= 1

    return masked_image

def random_masked_channels(image_batch, mask_kernel_size, slice_size, binary_mask, mask_rate, is_random=False):
    """
        为图像批次创建遮蔽区域。
        :param method:
        :param image_batch: ndarray, 图像的批次，形状为 (slice_num, height, width)
        :param mask_kernel_size: int, 遮蔽块的尺寸
        :param slice_size: tuple, 每个子图的尺寸 (height, width)
        :param binary_mask: str, 四位二进制字符串，表示哪些区域需要遮蔽
        :param mask_rate: float, 遮蔽的比例
        :return: ndarray, 遮蔽后的图像批次
        """
    pro_size = [3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    binary_mask_random = ['1000', '0100', '0010', '0001']
    slice_binary_mask = binary_mask
    slice_num = image_batch.shape[0]
    # 初始化原图的遮蔽掩码

    masked_image = np.zeros(image_batch.shape)

    # 检查每一子图是否需要遮蔽
    # for index in range(batch_size):
    for slice_index in range(slice_num):

        if is_random:
            mask_kernel_size = pro_size[random.randint(0, 10)]
            if random.uniform(0.0, 1.0) > 0.8:
                slice_binary_mask = binary_mask_random[random.randint(0, 3)]
                mask_rate = random.uniform(0.75, 1.0)
            else :
                mask_rate = random.uniform(0.0, 1.0)
        else:
            slice_binary_mask = binary_mask
        # 构建遮蔽块计数图的维度
        masked_sub_image_size = slice_size // mask_kernel_size
        # print(mask_rate, mask_kernel_size, slice_binary_mask)
        masked_sub_image_count = np.zeros((masked_sub_image_size, masked_sub_image_size), dtype=bool)



        for i in range(4):
            if slice_binary_mask[i] == '0':
                masked_image[slice_index, i, :, :] = 1
                continue
                # 计算子图的位置


            num_blocks = masked_sub_image_count.size
            num_keep = math.ceil(num_blocks * (1 - mask_rate))

            # 获取当前未被选择的块索引
            available_blocks = np.argwhere(masked_sub_image_count == 0)
            np.random.shuffle(available_blocks)

            keep_blocks = available_blocks[:num_keep]

            channel_mask = torch.zeros((slice_size, slice_size), dtype=torch.float32)
            for block in keep_blocks:
                block_h, block_w = block
                if mask_rate >= 0.75:
                    masked_sub_image_count[block_h, block_w] = 1

                h_start = block_h * mask_kernel_size
                h_end = h_start + mask_kernel_size
                w_start = block_w * mask_kernel_size
                w_end = w_start + mask_kernel_size

                # 设置遮蔽块为0
                channel_mask[h_start:h_end, w_start:w_end] = 1

            masked_image[slice_index, i, :, :] = channel_mask

    return masked_image
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

    image_batch = np.random.rand(1, 4, 192, 192).astype(np.float32)
    # image_shape = (6, 6)  # 原图大小
    mask_kernel_size = 3  # 遮蔽块大小
    single_image_size = 192  # 每个子图的大小
    binary_mask = '1000'  # 指定遮蔽的区域
    mask_rate = 1  # 遮蔽的比例
    method = 'channels'
    start = time.time()
    # masked_image = random_masked_channels(image_batch, mask_kernel_size, single_image_size, binary_mask, mask_rate, is_random=False)
    masked_image = random_masked_channels(image_batch, mask_kernel_size, single_image_size, binary_mask, mask_rate, is_random=True)
    end = time.time()

    masked_result = image_batch * masked_image
    print(f"mux time taken: {end - start:.4f} seconds")
    if method == 'channels':
        # 设置图像显示的大小
        plt.figure(figsize=(20, 5))
        # 显示 masked_image
        plt.subplot(1, 4, 1)  # 1行3列的第1个位置
        plt.imshow(masked_image[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('T1ce mask')  # 添加标题
        plt.axis('off')  # 关闭坐标轴显示

        # 显示 masked_result
        plt.subplot(1, 4, 2)  # 1行3列的第2个位置
        plt.imshow(masked_image[0, 1, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('T1n mask')
        plt.axis('off')  # 关闭坐标轴显示

        # 显示原始 images
        plt.subplot(1, 4, 3)  # 1行3列的第3个位置
        plt.imshow(masked_image[0, 2, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('T2w mask')
        plt.axis('off')  # 关闭坐标轴显示
        # 显示原始 images
        plt.subplot(1, 4, 4)  # 1行3列的第3个位置
        plt.imshow(masked_image[0, 3, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('T2f mask')
        plt.axis('off')  # 关闭坐标轴显示
    elif method == 'plane':
        # 设置图像显示的大小
        plt.figure(figsize=(15, 5))
        # 显示 masked_image
        plt.subplot(1, 3, 1)  # 1行3列的第1个位置
        plt.imshow(masked_image[0, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('Masked Image')  # 添加标题
        plt.axis('off')  # 关闭坐标轴显示

        # 显示 masked_result
        plt.subplot(1, 3, 2)  # 1行3列的第2个位置
        plt.imshow(masked_result[0, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('Masked Result')
        plt.axis('off')  # 关闭坐标轴显示

        # 显示原始 images
        plt.subplot(1, 3, 3)  # 1行3列的第3个位置
        plt.imshow(image_batch[0, :, :], cmap='gray', vmin=0, vmax=1)
        plt.colorbar()  # 添加颜色条
        plt.title('Original Image')
        plt.axis('off')  # 关闭坐标轴显示

    # 显示整个图
    plt.show()

    print(masked_image.shape)
