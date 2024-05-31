import datetime
import os
from pathlib import Path

import torch

from tqdm import tqdm
from utils import Logger, TensorboardLogger, create_checkpoint

from datasets import get_brats_dataloader
from utils.convert_shape import swap_batch_slice_dimensions, delete_batch_dimensions
from mask_generator import random_masked_area


def train(config, net, device, criterion, optimizer_f, scheduler_f, metric, resume, concat_method='plane'):
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


    # 训练数据集与验证数据集
    brats_train_root = config['data']['train']
    brats_valid_root = config['data']['valid']
    train_loader = get_brats_dataloader(root_dir=brats_train_root, batch_size=batch_size, slice_deep=slice_deep,
                                        slice_size=slice_size,
                                        mask_kernel_size=mask_kernel_size, binary_mask=train_binary_mask,
                                        mask_rate=train_mask_rate,
                                        num_workers=6, mode='train', concat_method=concat_method)
    valid_loader = get_brats_dataloader(root_dir=brats_valid_root, batch_size=batch_size, slice_deep=slice_deep,
                                        slice_size=slice_size,
                                        mask_kernel_size=mask_kernel_size, binary_mask=test_binary_mask,
                                        mask_rate=test_mask_rate,
                                        num_workers=4, mode='valid', concat_method=concat_method)

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
        'concat': concat_method,
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
        'save_root': save_root,
        'model': config['train']['model'],
        'optimizer': config['train']['optimizer'],
        'lr': config['train']['learning_rate'],
        'scheduler': config['train']['scheduler'],
    }
    logger_fac.log_config(training_settings)

    # 创建 TensorBoard 记录器
    tb_logger = TensorboardLogger(save_root)

    # 训练网络
    logger_fac.info("Training Start")

    # 训练所需的所有2D图像数量=训练数据集长度（人数） * 剪裁后的切片数量
    epoch_slice = train_loader.dataset.__len__() * slice_deep
    # 每个epoch包含的step数量
    step_per_epoch = slice_deep // step_slice

    if resume:
        start_epoch = config['train']['last_epoch']
        best_loss = config['train']['best_loss']
        # optimizer_f.optimizer.param_groups[0]['lr'] = config['train']['last_learning_rate']
        logger_fac.info(f"Resuming training from epoch {start_epoch}"
                        f" with best_loss {best_loss} and learning rate {optimizer_f.optimizer.param_groups[0]['lr']}")
    else:
        start_epoch = 0
        best_loss = float('inf')
        logger_fac.info("New Model")
    logger_fac.info("---------------------------------------------------------------------------------------------")

    # 已处理的step数量
    processed_step = 0
    # 总共的step数量
    total_step = step_per_epoch * train_loader.dataset.__len__()
    for epoch in range(start_epoch, epochs):
        # 确保模型处于训练模式
        net.train()
        running_loss = 0.0
        # avg_loss = 0.0
        # 计数当前epoch的step数量，每个epoch清零
        epoch_processed_step = 0
        epoch_processed_slices = 0
        logger_f.info(f"Epoch: {epoch + 1} - Current lr: {optimizer_f.get_learning_rate()}")
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch_person") as pbar:
            for masked_images, original_images in pbar:
                # 将数据和标签移动到设备上
                # 交换维度Batch_size和slice_size
                # 将slice_size作为真实的Batch_size
                # Batch_size设置为1，交换后代表单通道图像)
                if concat_method == 'plane':
                    masked_images = swap_batch_slice_dimensions(masked_images).to(device)
                    original_images = swap_batch_slice_dimensions(original_images).to(device)
                elif concat_method == 'channels':
                    masked_images = delete_batch_dimensions(masked_images).to(device)
                    original_images = delete_batch_dimensions(original_images).to(device)
                # 每个epoch下的step
                # step的数量=一个人总切片数量 // 每次step训练的切片数量

                # 默认整除
                for step in range(step_per_epoch):
                    masked_images_step = masked_images[range(step, masked_images.shape[0],
                                                             step_per_epoch), :, :, :]
                    original_images_step = original_images[range(step, original_images.shape[0],
                                                                 step_per_epoch), :, :, :]

                    # 清空之前的梯度
                    optimizer_f.zero_grad()

                    # 前向传播
                    outputs = net(masked_images_step)

                    # 计算损失
                    # loss_value = criterion.calculate_loss_regions(outputs, original_images_step, binary_masks=train_binary_mask)
                    loss_value = criterion.calculate_loss_regions(outputs, original_images_step,
                                                                  binary_masks=train_binary_mask)

                    # 反向传播
                    loss_value.backward()

                    # 更新模型参数
                    optimizer_f.step()

                    # 累计损失并显示批次损失均值
                    running_loss += loss_value.item()
                    # 更新进度条描述
                    avg_loss = running_loss / (epoch_processed_step + 1)  # 计算当前平均损失
                    # 更新进度变量
                    epoch_processed_step += 1
                    epoch_processed_slices += step_slice
                    # 更新进度条
                    pbar.set_description(
                        f"Epoch {epoch + 1}/{epochs}; Step: {epoch_processed_step}/{total_step}; "
                        f"Slice: {epoch_processed_slices}/{epoch_slice} ")
                    pbar.set_postfix(loss=avg_loss)  # 显示当前step的平均损失

                    # 更新总Step
                    processed_step += 1
                    # 记录训练损失到 TensorBoard
                    logger_f.info(f"Step: {processed_step} - Train/Loss: {avg_loss}")
                    tb_logger.log_scalar('Train/Loss', avg_loss, processed_step)
            pbar.update()
            # 调整学习率
            scheduler_f.step()

        # 验证模型性能
        net.eval()  # 设置模型为评估模式
        valid_loss = 0.0
        avg_psnr = [0.0] * 4
        avg_ssim = [0.0] * 4
        count = 0

        torch.cuda.empty_cache()
        with torch.no_grad():  # 关闭梯度计算
            with tqdm(valid_loader, desc="Validation", unit="batch_person") as pbar_test:
                for masked_images, original_images in pbar_test:
                    if concat_method == 'plane':
                        masked_images = swap_batch_slice_dimensions(masked_images).to(device)
                        original_images = swap_batch_slice_dimensions(original_images).to(device)
                    elif concat_method == 'channels':
                        masked_images = delete_batch_dimensions(masked_images).to(device)
                        original_images = delete_batch_dimensions(original_images).to(device)
                    for step in range(slice_deep // step_slice):
                        masked_images_step = masked_images[range(step, masked_images.shape[0],
                                                                 step_per_epoch), :, :, :]
                        original_images_step = original_images[range(step, original_images.shape[0],
                                                                     step_per_epoch), :, :, :]
                        outputs = net(masked_images_step)

                        # 计算损失
                        # test_loss += criterion.calculate_loss_regions(outputs, original_images_step,
                        #                                               binary_masks=test_binary_mask)
                        valid_loss += criterion.calculate_loss_regions(outputs, original_images_step,
                                                                       binary_masks=test_binary_mask)
                        # 计算 PSNR 和 SSIM
                        current_psnr, current_ssim = metric(outputs, original_images_step, binary_masks=test_binary_mask,
                                                            concat_method=concat_method)
                        # 累加每个象限的 PSNR 和 SSIM
                        for j in range(4):
                            avg_psnr[j] += current_psnr[j]
                            avg_ssim[j] += current_ssim[j]
                        count += 1
                pbar_test.update()

        valid_loss /= len(valid_loader)
        avg_psnr_total = [x / count for x in avg_psnr]
        avg_ssim_total = [x / count for x in avg_ssim]

        # 打印结果和写入信息
        logger_fac.info(f"Validation/Loss: {valid_loss:.4f}")
        tb_logger.log_scalar('Validation/Loss', valid_loss, epoch)
        # print("平均 PSNR: ", " ".join([f"{x:.4f}" for x in avg_psnr_total]))
        # print("平均 SSIM: ", " ".join([f"{x:.4f}" for x in avg_ssim_total]))

        # 打印每种模态的详细 PSNR 和 SSIM
        psnr_message = (f"Validation/PSNR "
                        f"T1c: {avg_psnr_total[0]:.4f}, T1n: {avg_psnr_total[1]:.4f}, "
                        f"T2w: {avg_psnr_total[2]:.4f}, T2f: {avg_psnr_total[3]:.4f}")
        ssim_message = (f"Validation/SSIM "
                        f"T1c: {avg_ssim_total[0]:.4f}, T1n: {avg_ssim_total[1]:.4f}, "
                        f"T2w: {avg_ssim_total[2]:.4f}, T2f: {avg_ssim_total[3]:.4f}")
        logger_fac.info(psnr_message)
        logger_fac.info(ssim_message)
        tb_logger.log_scalar('Validation/PSNR_T1c', avg_psnr_total[0], epoch)
        tb_logger.log_scalar('Validation/PSNR_T1n', avg_psnr_total[1], epoch)
        tb_logger.log_scalar('Validation/PSNR_T2w', avg_psnr_total[2], epoch)
        tb_logger.log_scalar('Validation/PSNR_T2f', avg_psnr_total[3], epoch)

        tb_logger.log_scalar('Validation/SSIM_T1c', avg_ssim_total[0], epoch)
        tb_logger.log_scalar('Validation/SSIM_T1n', avg_ssim_total[1], epoch)
        tb_logger.log_scalar('Validation/SSIM_T2w', avg_ssim_total[2], epoch)
        tb_logger.log_scalar('Validation/SSIM_T2f', avg_ssim_total[3], epoch)

        # print( f"验证 SSIM T1c {avg_ssim_total[0]:.4f}, T1n {avg_ssim_total[1]:.4f}, T2w {avg_ssim_total[2]:.4f},
        # T2f {avg_ssim_total[3]:.4f}")

        # 保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            file_name = f'best_model_epoch_{epoch + 1}.ckpt'
            best_model_path = save_root / file_name
            create_checkpoint(epoch + 1, net, optimizer_f, scheduler_f, valid_loss, best_model_path)
            logger_fac.info(f"Saved best model at epoch {epoch + 1} to {best_model_path}")

        # 保存定期的检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_root / f'checkpoint_epoch_{epoch + 1}.ckpt'
            create_checkpoint(epoch + 1, net, optimizer_f, scheduler_f, valid_loss, checkpoint_path)
            logger_fac.info(f"Saved checkpoint at epoch {epoch + 1}")
        logger_fac.info("---------------------------------------------------------------------------------------------")
        torch.cuda.empty_cache()
    # 关闭 TensorBoard 记录器
    tb_logger.close()
