from datasets import get_brats_dataloader
from evaluations.metrics import *
from utils import show_mask_origin, Logger
from tqdm import tqdm

from utils.convert_shape import swap_batch_slice_dimensions, delete_batch_dimensions


def extract_region(img, quadrant, size):
    """
    根据象限从图像中提取区域。
    :param img: 输入图像 [height, width]
    :param quadrant: 象限编号 (1, 2, 3, 4)
    :param size: 每个区域的尺寸 (height, width)
    :return: 提取的区域
    """
    h, w = size
    if quadrant == 0:
        return img[:h, :w]  # 左上
    elif quadrant == 1:
        return img[:h, -w:]  # 右上
    elif quadrant == 2:
        return img[-h:, :w]  # 左下
    elif quadrant == 3:
        return img[-h:, -w:]  # 右下


def calculate_metrics(target, ref, device='cuda', binary_masks=None, concat_method='plane'):
    """
    计算模型在给定输入和目标之间的性能指标。
    :param concat_method:
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
        # ssim_func = ssim
    else:
        psnr_func = psnr_np
        # ssim_func = ssim_np

    # 遍历所有batch和slice
    for i in range(batch_size):
        for area in range(4):
            if concat_method == 'plane':
                # 提取对应象限的target和output区域
                target_region = extract_region(ref[i, 0, :, :], area, region_size)
                output_region = extract_region(target[i, 0, :, :], area, region_size)
            elif concat_method == 'channels':
                target_region = ref[i, area, :, :]
                output_region = target[i, area, :, :]
            else:
                raise NotImplementedError("Invalid concat_method")

            # 调整维度以适应ssim函数要求
            target_region = target_region.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            output_region = output_region.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # 计算PSNR和SSIM
            cur_psnr = psnr_func(target_region, output_region)
            # cur_ssim = ssim_func(target_region, output_region)

            avg_psnr[area] += cur_psnr
            # avg_ssim[quadrant - 1] += cur_ssim

    # 计算平均值
    avg_psnr = [x / batch_size for x in avg_psnr]
    # avg_ssim = [x / batch_size for x in avg_ssim]

    # return avg_psnr, avg_ssim
    return avg_psnr


def evaluation(config, net, device, criterion, show_image=False, concat_method='plane'):
    # 剪裁后切片的数量
    slice_deep = config['train']['slice_deep']
    # 剪裁后切片的宽高尺寸
    slice_size = config['train']['slice_size']
    # 批次大小，默认为1
    batch_size = config['train']['batch_size']
    # 每步使用的切片数量，默认小于slice_deep
    step_slice = config['train']['step_slice']

    # 遮蔽块的宽高尺寸
    mask_kernel_size = config['mask']['mask_kernel_size']
    # 测试时候的遮蔽选项
    test_binary_mask = config['mask']['test_binary_mask']
    # 测试时候的遮蔽率
    test_mask_rate = config['mask']['test_mask_rate']

    brats_test_root = config['data']['test']

    test_loader = get_brats_dataloader(root_dir=brats_test_root, batch_size=batch_size, slice_deep=slice_deep,
                                       slice_size=slice_size,
                                       mask_kernel_size=mask_kernel_size, binary_mask=test_binary_mask,
                                       mask_rate=test_mask_rate,
                                       num_workers=2, mode='test', concat_method=concat_method)
    logger_c = Logger(None, dst='console')

    # 每个epoch包含的step数量
    step_per_epoch = slice_deep // step_slice
    index = [0, 8, 15]
    # 验证模型性能
    net.eval()  # 设置模型为评估模式
    test_loss = 0.0
    avg_psnr = [0.0] * 4
    # avg_ssim = [0.0] * 4
    count = 0
    loop = 3
    torch.cuda.empty_cache()
    with torch.no_grad():  # 关闭梯度计算
        with tqdm(test_loader, desc="Validation", unit="batch_person") as pbar_test:
            for masked_images, original_images in pbar_test:
                if concat_method == 'plane':
                    masked_images = swap_batch_slice_dimensions(masked_images).to(device)
                    original_images = swap_batch_slice_dimensions(original_images).to(device)
                elif concat_method == 'channels':
                    masked_images = delete_batch_dimensions(masked_images).to(device)
                    original_images = delete_batch_dimensions(original_images).to(device)
                for step in range(step_per_epoch):
                    masked_images_step = masked_images[range(step, masked_images.shape[0],
                                                             step_per_epoch), :, :, :]
                    original_images_step = original_images[range(step, original_images.shape[0],
                                                                 step_per_epoch), :, :, :]
                    outputs = net(masked_images_step)

                    # 计算损失
                    test_loss += criterion.calculate_loss_regions(outputs, original_images_step,
                                                                  binary_masks=test_binary_mask)
                    # test_loss = criterion.calculate_loss_no_background(outputs, original_images_step)
                    # 计算 PSNR 和 SSIM
                    current_psnr = calculate_metrics(outputs, original_images_step, binary_masks=test_binary_mask,
                                                     concat_method=concat_method)
                    if show_image and loop > 0:
                        show_mask_origin(outputs, masked_images_step, original_images_step, index,
                                         concat_method=concat_method)
                        loop -= 1
                    # 累加每个象限的 PSNR 和 SSIM
                    for j in range(4):
                        avg_psnr[j] += current_psnr[j]
                    count += 1
            pbar_test.update()

    test_loss /= len(test_loader)
    avg_psnr_total = [x / count for x in avg_psnr]
    # avg_ssim_total = [x / count for x in avg_ssim]

    # 打印结果和写入信息
    logger_c.info(f"Test/Loss: {test_loss:.4f}")

    # 打印每种模态的详细 PSNR 和 SSIM
    psnr_message = (f"Test/PSNR "
                    f"T1c: {avg_psnr_total[0]:.4f}, T1n: {avg_psnr_total[1]:.4f}, "
                    f"T2w: {avg_psnr_total[2]:.4f}, T2f: {avg_psnr_total[3]:.4f}")
    logger_c.info(psnr_message)



