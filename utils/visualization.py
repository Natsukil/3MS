from dataset_conversion.BraTsData_person import Dataset_brats, get_brats_dataloader
from utils.swap_dimensions import swap_batch_slice_dimensions
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

root_dir = "E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"


def get_dataloader_iterator():
    dataloader = get_brats_dataloader(slice_size=100, root_dir=root_dir, batch_size=1, shuffle=False, num_workers=4,
                                      mask_rate=1, binary_mask='1000', mode='eval')
    return iter(dataloader)

def show_mask_origin(X, y, index):

    print("original:", X.shape, y.shape)

    # X_new, y_new = swap_batch_slice_dimensions(X), swap_batch_slice_dimensions(y)
    # print("new:", X_new.shape, y_new.shape)
    # 设置图像显示的大小
    plt.figure(figsize=(10, 5))

    # 显示 masked_image
    plt.subplot(1, 2, 1)  # 1行3列的第1个位置
    plt.imshow(X[index, 0, :, :], cmap='gray')
    plt.colorbar()  # 添加颜色条
    plt.title('Masked Image')  # 添加标题
    plt.axis('off')  # 关闭坐标轴显示

    # 显示 masked_result
    plt.subplot(1, 2, 2)  # 1行3列的第2个位置
    plt.imshow(y[index, 0, :, :], cmap='gray')
    plt.colorbar()  # 添加颜色条
    plt.title('Masked Result')
    plt.axis('off')  # 关闭坐标轴显示
    # 显示 mask
    # plt.subplot(1, 2, 2)  # 1行3列的第2个位置
    # plt.imshow(mask[50, 0, :, :], cmap='gray')
    # plt.colorbar()  # 添加颜色条
    # plt.title('Masked Result')
    # plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def calculate_mutil_model_pixel_value():
    dataloader = get_brats_dataloader(root_dir, batch_size=1, num_workers=8)
    data_iter = iter(dataloader)
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
    # print(f"Minimum values across the dataset:\n T1: {t1_min}\n T2c: {t2c_min}\n T2f: {t2f_min}\n FLAIR: {flair_min}")


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
def test_eval_regions(X, y, index):
    batch_size, channels, height, width = X.shape
    region_size = (height//2, width//2)
    X_new, y_new = swap_batch_slice_dimensions(X), swap_batch_slice_dimensions(y)
    regions = {
        't1c': extract_region(y_new[index, 0, :, :], 1, region_size),  # 左上
        't1n': extract_region(y_new[index, 0, :, :], 2, region_size),  # 右上
        't2w': extract_region(y_new[index, 0, :, :], 3, region_size),  # 左下
        't2f': extract_region(y_new[index, 0, :, :], 4, region_size),  # 右下
    }
    plt.figure(figsize=(10, 5))

    for idx, (region_name, region) in enumerate(regions.items(), 1):
        plt.subplot(2, 2, idx)
        plt.imshow(region, cmap='gray')
        plt.colorbar()
        plt.title(f'{region_name} Region')
        plt.axis('off')

    plt.suptitle(f'Visualizing Different Regions of a Single Batch in evaluationc: Slice {index}')
    plt.show()

if __name__ == '__main__':
    # 获取迭代器
    data_iter = get_dataloader_iterator()
    X, y = next(data_iter)  # 获取一个批次的图像
    index_show=99
    # 显示图片
    show_mask_origin(X, y,index_show)
    # 计算像素值
    # calculate_mutil_model_pixel_value()
    # 测试切分片
    test_slice_result(X, y,index_show)
    test_eval_regions(X, y,index_show)
