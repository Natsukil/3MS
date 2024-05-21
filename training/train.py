import datetime
import os
from pathlib import Path

import torch

from tqdm import tqdm
from utils import Logger, TensorboardLogger

from datasets import get_brats_dataloader
from utils.swap_dimensions import swap_batch_slice_dimensions


def train(config, net, device, criterion, optimizer, scheduler, metric):
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
    # 训练时的遮蔽选项
    train_binary_mask = config['mask']['train_binary_mask']
    # 测试时候的遮蔽选项
    test_binary_mask = config['mask']['test_binary_mask']
    # 训练时的遮蔽率
    train_mask_rate = config['mask']['train_mask_rate']
    # 测试时候的遮蔽率
    test_mask_rate = config['mask']['test_mask_rate']

    # 训练轮数
    epochs = config['train']['epochs']

    # 定义模型保存路径
    save_root = "result/models/" + config['train']['model']
    current_time = datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")
    save_root = Path(save_root) / (train_binary_mask + current_time)
    os.makedirs(save_root, exist_ok=True)  # 创建目录
    # model_path = Path(model_save_dir)/ 'best.ckpt'

    # 设置日志
    logger_fac = Logger(save_root, dst='both')
    logger_f = Logger(save_root, dst='file')
    training_settings = {
        'slice_deep': slice_deep,
        'slice_size': slice_size,
        'batch_size': batch_size,
        'step_slice': step_slice,
        'mask_kernel_size': mask_kernel_size,
        'train_binary_mask': train_binary_mask,
        'test_binary_mask': test_binary_mask,
        'train_mask_rate': train_mask_rate,
        'test_mask_rate': test_mask_rate,
        'epochs': epochs,
        'device': device,
        'save_root': save_root
    }
    logger_fac.log_config(training_settings)

    # 创建 TensorBoard 记录器
    tb_logger = TensorboardLogger(save_root)

    # 训练数据集与测试数据集
    brats_train_root = config['data']['train']
    brats_test_root = config['data']['test']
    train_loader = get_brats_dataloader(root_dir=brats_train_root, batch_size=batch_size, slice_deep=slice_deep,
                                        slice_size=slice_size,
                                        mask_kernel_size=mask_kernel_size, binary_mask=train_binary_mask,
                                        mask_rate=train_mask_rate,
                                        num_workers=1, mode='train')
    test_loader = get_brats_dataloader(root_dir=brats_test_root, batch_size=batch_size, slice_deep=slice_deep,
                                       slice_size=slice_size,
                                       mask_kernel_size=mask_kernel_size, binary_mask=test_binary_mask,
                                       mask_rate=test_mask_rate,
                                       num_workers=4, mode='eval')

    # 初始化用于跟踪最佳模型的变量
    # 最优损失函数
    best_loss = float('inf')
    # 训练网络
    logger_fac.info("Training Start")
    logger_fac.info("-----------------------------------------------------------------------------------------------")

    # 训练所需的所有2D图像数量=训练数据集长度（人数） * 剪裁后的切片数量
    epoch_slice = train_loader.dataset.__len__() * slice_deep
    # 每个epoch包含的step数量
    step_per_epoch = slice_deep // step_slice
    # 已处理的切片数量
    processed_slices = 0
    # 总共的step数量
    total_step = step_per_epoch * train_loader.dataset.__len__()
    for epoch in range(epochs):
        # 确保模型处于训练模式
        net.train()
        running_loss = 0.0
        avg_loss = 0.0
        # 计数当前epoch的step数量，每个epoch清零
        processed_step = 0
        logger_f.info(f"Epoch: {epoch + 1} - Current lr: {optimizer.get_learning_rate()}")
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch_person") as pbar:
            for masked_images, original_images in pbar:
                # 将数据和标签移动到设备上
                # 交换维度Batch_size和slice_size
                # 将slice_size作为真实的Batch_size
                # Batch_size设置为1，交换后代表单通道图像)
                masked_images = swap_batch_slice_dimensions(masked_images).to(device)
                original_images = swap_batch_slice_dimensions(original_images).to(device)
                # 每个epoch下的step
                # step的数量=一个人总切片数量 // 每次step训练的切片数量
                # 默认整除
                for step in range(step_per_epoch):
                    masked_images_step = masked_images[range(step, masked_images.shape[0],
                                                             step_per_epoch), :, :, :]
                    original_images_step = original_images[range(step, original_images.shape[0],
                                                                 step_per_epoch), :, :, :]

                    # 清空之前的梯度
                    optimizer.zero_grad()

                    # 前向传播
                    outputs = net(masked_images_step)

                    # 计算损失
                    loss_value = criterion.calculate_loss(outputs, original_images_step, binary_masks=train_binary_mask)

                    # 反向传播
                    loss_value.backward()

                    # 更新模型参数
                    optimizer.step()

                    # 累计损失并显示批次损失均值
                    running_loss += loss_value.item()
                    # 更新进度条描述
                    avg_loss = running_loss / (processed_step + 1)  # 计算当前平均损失
                    # 更新进度变量
                    processed_step += 1
                    processed_slices += step_slice
                    # 更新进度条
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{epochs}; Step: {processed_step}/{total_step}; "
                        f"Slice: {processed_slices}/{epoch_slice} ")
                    pbar.set_postfix(loss=avg_loss)  # 显示当前step的平均损失

                    # 记录训练损失到 TensorBoard
                    logger_f.info(f"Step: {processed_step} - Train/Loss: {avg_loss}")
                    tb_logger.log_scalar('Train/Loss', avg_loss, processed_step)
            pbar.update()
            # 调整学习率
            scheduler.step()

        # 验证模型性能
        net.eval()  # 设置模型为评估模式
        test_loss = 0.0
        avg_psnr = [0.0] * 4
        # avg_ssim = [0.0] * 4
        count = 0

        torch.cuda.empty_cache()
        with torch.no_grad():  # 关闭梯度计算
            with tqdm(test_loader, desc="Validation", unit="batch_person") as pbar_test:
                for masked_images, original_images in pbar_test:
                    masked_images = swap_batch_slice_dimensions(masked_images).to(device)
                    original_images = swap_batch_slice_dimensions(original_images).to(device)
                    for step in range(slice_deep // step_slice):
                        outputs = net(masked_images)

                        # 计算损失
                        test_loss += criterion.calculate_loss(outputs, original_images,
                                                              binary_masks=test_binary_mask).item()
                        # 计算 PSNR 和 SSIM
                        current_psnr = metric(outputs, original_images, binary_masks=test_binary_mask)
                        # 累加每个象限的 PSNR 和 SSIM
                        for j in range(4):
                            avg_psnr[j] += current_psnr[j]
                            # avg_ssim[j] += current_ssim[j]
                        count += 1
                pbar_test.update()

        test_loss /= len(test_loader)
        avg_psnr_total = [x / count for x in avg_psnr]
        # avg_ssim_total = [x / count for x in avg_ssim]

        # 打印结果和写入信息
        logger_fac.info(f"Validation/Loss: {test_loss:.4f}")
        tb_logger.log_scalar('Validation/Loss', test_loss, epoch)
        # print("平均 PSNR: ", " ".join([f"{x:.4f}" for x in avg_psnr_total]))
        # print("平均 SSIM: ", " ".join([f"{x:.4f}" for x in avg_ssim_total]))

        # 打印每种模态的详细 PSNR 和 SSIM
        psnr_message = (f"Validation/PSNR "
                        f"T1c: {avg_psnr_total[0]:.4f}, T1n: {avg_psnr_total[1]:.4f}, "
                        f"T2w: {avg_psnr_total[2]:.4f}, T2f: {avg_psnr_total[3]:.4f}")
        logger_fac.info(psnr_message)
        tb_logger.log_scalar('Validation/PSNR_T1c', avg_psnr_total[0], processed_step)
        tb_logger.log_scalar('Validation/PSNR_T1n', avg_psnr_total[1], processed_step)
        tb_logger.log_scalar('Validation/PSNR_T2w', avg_psnr_total[2], processed_step)
        tb_logger.log_scalar('Validation/PSNR_T2f', avg_psnr_total[3], processed_step)

        # print( f"验证 SSIM T1c {avg_ssim_total[0]:.4f}, T1n {avg_ssim_total[1]:.4f}, T2w {avg_ssim_total[2]:.4f},
        # T2f {avg_ssim_total[3]:.4f}")

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            file_name = f'best_model_epoch_{epoch + 1}.ckpt'
            best_model_path = save_root / file_name
            torch.save(net.state_dict(), best_model_path)
            logger_fac.info(f"Saved best model at epoch {epoch + 1} to {best_model_path}")

        # 保存定期的检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_root / f'checkpoint_epoch_{epoch + 1}.ckpt'
            torch.save(net.state_dict(), checkpoint_path)
            logger_fac.info(f"Saved checkpoint at epoch {epoch + 1}")
        logger_fac.info("-----------------------------------------------------------------------------------------------")
        torch.cuda.empty_cache()
    # args = get_args()
    # 关闭 TensorBoard 记录器
    tb_logger.close()
