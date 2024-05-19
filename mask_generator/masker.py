import random
import numpy as np
# from dataset_conversion import BraTsData


def random_masked_area(image_batch, mask_kernel_size, slice_size, binary_mask, mask_rate):
    """
    为图像批次创建遮蔽区域。

    :param image_batch: ndarray, 图像的批次，形状为 (slice_num, height, width)
    :param mask_kernel_size: int, 遮蔽块的尺寸
    :param slice_size: tuple, 每个子图的尺寸 (height, width)
    :param binary_mask: str, 四位二进制字符串，表示哪些区域需要遮蔽
    :param mask_rate: float, 遮蔽的比例
    :return: ndarray, 遮蔽后的图像批次
    """
    slice_num, height, width = image_batch.shape
    sub_height, sub_width = slice_size.slice_size
    # 初始化原图的遮蔽掩码
    masked_image = np.ones(image_batch.shape)



    # 检查每一子图是否需要遮蔽
    # for index in range(batch_size):
    for slice_index in range(slice_num):

        # 构建遮蔽块计数图的维度
        masked_sub_image_width = slice_size // mask_kernel_size
        masked_sub_image_height = sub_height // mask_kernel_size
        masked_sub_image_count = np.zeros((masked_sub_image_width, masked_sub_image_height))

        for i in range(4):
            if binary_mask[i] == '0':
                continue
            # 计算子图的位置
            row_start = (i // 2) * sub_height
            row_end = row_start + sub_height
            col_start = (i % 2) * sub_width
            col_end = col_start + sub_width

            # 收集所有可以被选择为遮蔽块的位置
            all_blocks = [(r, c) for r in range(0, sub_height, mask_kernel_size)
                          for c in range(0, sub_width, mask_kernel_size)
                          if masked_sub_image_count[r // mask_kernel_size, c // mask_kernel_size] < 3]

            random.shuffle(all_blocks)  # 随机化块的顺序

            # 根据rate计算需要遮蔽的块数
            mask_count = int(mask_rate * len(all_blocks))

            # 选择块并设置遮蔽
            selected_blocks = all_blocks[:int(mask_count)]
            for r, c in selected_blocks:
                masked_image[slice_index, row_start + r:row_start + r + mask_kernel_size,
                col_start + c:col_start + c + mask_kernel_size] = 0  # 设置遮蔽块
                masked_sub_image_count[r // mask_kernel_size:(r // mask_kernel_size) + 1,
                c // mask_kernel_size:(c // mask_kernel_size) + 1] += 1

    return masked_image


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     dataloader = BraTsData.get_dataloader("E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
#                                           batch_size=1)
#     data_iter = iter(dataloader)
#     images = next(data_iter)  # 获取一个批次的图像
#
#     # image_shape = (6, 6)  # 原图大小
#     mask_kernel_size = 12  # 遮蔽块大小
#     single_image_size = (192, 192)  # 每个子图的大小
#     binary_mask = '1111'  # 指定遮蔽的区域
#     mask_rate = 0.5  # 遮蔽的比例
#
#     masked_image = random_masked_area(images, mask_kernel_size, single_image_size, binary_mask, mask_rate)
#     masked_result = np.where(masked_image == 1, images, -1)
#
#     # 设置图像显示的大小
#     plt.figure(figsize=(15, 5))
#
#     # 显示 masked_image
#     plt.subplot(1, 3, 1)  # 1行3列的第1个位置
#     plt.imshow(masked_image[0, 2, :, :], cmap='gray')
#     plt.colorbar()  # 添加颜色条
#     plt.title('Masked Image')  # 添加标题
#     plt.axis('off')  # 关闭坐标轴显示
#
#     # 显示 masked_result
#     plt.subplot(1, 3, 2)  # 1行3列的第2个位置
#     plt.imshow(masked_result[0, 50, :, :], cmap='gray')
#     plt.colorbar()  # 添加颜色条
#     plt.title('Masked Result')
#     plt.axis('off')  # 关闭坐标轴显示
#
#     # 显示原始 images
#     plt.subplot(1, 3, 3)  # 1行3列的第3个位置
#     plt.imshow(images[0, 50, :, :], cmap='gray')
#     plt.colorbar()  # 添加颜色条
#     plt.title('Original Image')
#     plt.axis('off')  # 关闭坐标轴显示
#
#     # 显示整个图
#     plt.show()
#
#     print(masked_image.shape)
