from datasets import get_brats_dataloader
# from evaluations import extract_region
from utils.convert_shape import swap_batch_slice_dimensions
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def show_mask_origin(y_hat, X, y, index, concat_method='plane'):
    vmin = 0
    vmax = 1
    y_hat = y_hat.to('cpu')
    X = X.to('cpu')
    y = y.to('cpu')
    for i in index:
        if concat_method == 'plane':
            # 设置图像显示的大小
            plt.figure(figsize=(15, 5))

            # 显示 masked_image
            plt.subplot(1, 3, 1)  # 1行3列的第1个位置
            plt.imshow(y_hat[i, 0, :, :], cmap='gray', vmin=vmin, vmax=1)
            plt.colorbar()  # 添加颜色条
            plt.title(f'Y_hat index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            # 显示 masked_image
            plt.subplot(1, 3, 2)  # 1行3列的第1个位置
            plt.imshow(X[i, 0, :, :], cmap='gray', vmin=vmin, vmax=1)
            plt.colorbar()  # 添加颜色条
            plt.title(f'mask index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            # 显示 masked_result
            plt.subplot(1, 3, 3)  # 1行3列的第2个位置
            plt.imshow(y[i, 0, :, :], cmap='gray', vmin=vmin, vmax=1)
            plt.colorbar()  # 添加颜色条
            plt.title(f'Y index {i}')
            plt.axis('off')  # 关闭坐标轴显示
        elif concat_method == 'channels':
            # 获取切片数量和图像大小
            # slice_num, channels, w, h = y_hat.shape[1], y_hat.shape[2], y_hat.shape[3], y_hat.shape[4]

            # 设置图像显示的大小
            plt.figure(figsize=(10, 10))

            # 显示 y_hat
            plt.subplot(3, 4, 1)  # 3行3列的第1个位置
            plt.imshow(y_hat[i, 0, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y_hat t1c index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 2)  # 3行3列的第2个位置
            plt.imshow(y_hat[i, 1, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y_hat t1n index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 3)  # 3行3列的第3个位置
            plt.imshow(y_hat[i, 2, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y_hat t2w index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 4)  # 3行3列的第4个位置
            plt.imshow(y_hat[i, 3, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y_hat t2f index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            # 显示 X
            plt.subplot(3, 4, 5)  # 3行3列的第5个位置
            plt.imshow(X[i, 0, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'X t1c index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 6)  # 3行3列的第6个位置
            plt.imshow(X[i, 1, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'X t1n index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 7)  # 3行3列的第7个位置
            plt.imshow(X[i, 2, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'X t2w index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 8)  # 3行3列的第8个位置
            plt.imshow(X[i, 3, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'X t2f index {i}')  # 添加标题
            plt.axis('off')  # 关闭坐标轴显示

            # 显示 y
            plt.subplot(3, 4, 9)  # 3行3列的第9个位置
            plt.imshow(y[i, 0, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y t1c index {i}')
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 10)  # 3行3列的第10个位置
            plt.imshow(y[i, 1, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y t1n index {i}')
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 11)  # 3行3列的第11个位置
            plt.imshow(y[i, 2, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y t2w index {i}')
            plt.axis('off')  # 关闭坐标轴显示

            plt.subplot(3, 4, 12)  # 3行3列的第12个位置
            plt.imshow(y[i, 3, :, :], cmap='gray', vmin=vmin, vmax=1)
            # plt.colorbar()  # 添加颜色条
            plt.title(f'Y t2f index {i}')
            plt.axis('off')  # 关闭坐标轴显示
        plt.show()


def calculate_mutil_model_pixel_value(dataloader):
    # t1_min, t2c_min, t2f_min, flair_min = float('inf'),float('inf'),float('inf'),float('inf')
    global_min = float('inf')
    global_max = float('-inf')
    for X, y in tqdm(dataloader):
        # print(X.shape, y.shape)
        X_new, y_new = swap_batch_slice_dimensions(X), swap_batch_slice_dimensions(y)
        # print(X_new.shape, y_new.shape)

        # 获取当前批次的最小值/最大值
        current_global_min = torch.min(X_new)
        current_global_max = torch.max(X_new)
        # current_t1_min = torch.min(batch['t1n'])
        # current_t2c_min = torch.min(batch['t1c'])
        # current_t2f_min = torch.min(batch['t2w'])
        # current_flair_min = torch.min(batch['flair'])

        # 更新全局最小值/最大值
        global_min = min(current_global_min, global_min)
        global_max = max(current_global_max, global_max)
        # t1_min = min(t1_min, current_t1_min.item())
        # t2c_min = min(t2c_min, current_t2c_min.item())
        # t2f_min = min(t2f_min, current_t2f_min.item())
        # flair_min = min(flair_min, current_flair_min.item())
    print(f"Minimum values across the dataset:\n Global: {global_min}\n Global: {global_max}")
    # print(f"Minimum values across the datasets:\n T1: {t1_min}\n T2c: {t2c_min}\n T2f: {t2f_min}\n FLAIR: {flair_min}")


def test_slice_result(X, y, index):
    print("original:", X.shape, y.shape)

    X_new, y_new = swap_batch_slice_dimensions(X), swap_batch_slice_dimensions(y)
    print("new:", X_new.shape, y_new.shape)

    w, h = y.shape[2] // 2, y.shape[3] // 2

    # 从y_hat和y中提取四个区域
    regions = {
        't1c': (slice(None), slice(None), slice(0, w), slice(0, h)),  # 左上
        't1n': (slice(None), slice(None), slice(0, w), slice(h, 2 * h)),  # 右上
        't2w': (slice(None), slice(None), slice(w, 2 * w), slice(0, h)),  # 左下
        't2f': (slice(None), slice(None), slice(w, 2 * w), slice(h, 2 * h)),  # 右下
    }
    plt.figure(figsize=(10, 5))

    for idx, (region_name, slice_indices) in enumerate(regions.items(), 1):
        plt.subplot(2, 2, idx)
        region = y_new[index, 0, slice_indices[2], slice_indices[3]]  # 注意修改对应的X或y和切片索引
        plt.imshow(region, cmap='gray')
        plt.colorbar()
        plt.title(f'{region_name} Region')
        plt.axis('off')

    plt.suptitle(f'Visualizing Different Regions of a Single Batch: Slice{index}')
    plt.show()
