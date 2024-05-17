import argparse
from networks.UNet import UNet
from networks.S_UNet import S_UNet
import torch
import yaml
from init_weight import ModelInitializer
from dataset_conversion.BraTsData import Dataset_brats, get_dataloader
from loss_function import LossFunctions
from evaluations.eval import evaluate_model
import datetime
import os
from tqdm import tqdm
from utils.swap_dimensions import swap_batch_slice_dimensions
import logging
import time
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--save_dir', type=str, default='../checkpoints/', help='save directory')
    parser.add_argument('--load_dir', type=str, default='../checkpoints/UNet_BraTS2023_epoch_100.pth', help='load directory')
    parser.add_argument('--root_dir', type=str, default='../data/BraTS2023/', help='root directory of the dataset')
    parser.add_argument('--mask', type=str, default='config/UNet_2d_random_mask.yaml', help='config of mask')
    parser.add_argument('--dataset', type=str, default='BraTS2023', help='dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model', type=str, default='UNet', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--loss', type=str, default='DiceLoss', help='loss function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='learning rate scheduler')

    return parser.parse_args()


if __name__ == '__main__':
    batch_size = 1
    slice_size = 16
    learning_rate = 2e-4
    epochs = 100
    train_binary_mask = '1000'
    test_binary_mask = '1000'

    # 训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on:", device)

    # 定义网络模型
    net = S_UNet(in_channels=1, base_filters=32, bias=False)
    net.to(device)

    # 读取预训练模型，或从头训练
    pretrain = False
    load_dir = None
    if pretrain is True and load_dir is not None:
        net.load_state_dict(torch.load(load_dir))
        print(f"load model from {load_dir}")
    else:
        # 模型参数初始化
        initializer = ModelInitializer(method='xavier', uniform=False)
        initializer.initialize(net)
        print("Training a new model")

    # 定义模型保存路径
    save_dir = "result/models/S_UNet/"
    current_time = datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")
    directory = os.path.join(save_dir,train_binary_mask+current_time)
    os.makedirs(directory, exist_ok=True)  # 创建目录
    print("Save model in directory:", directory)
    # model_path = os.path.join(directory, 'best.ckpt')

    # 设置日志
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = current_time + ".log"
    logging.basicConfig(filename=os.path.join(log_dir, log_filename), level=logging.INFO)

    # 创建 TensorBoard 记录器
    tensorboard_dir = os.path.join(save_dir, "tensorboard_logs")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 定义损失函数
    loss = LossFunctions()

    # 定义性能评价指标
    eval = evaluate_model

    # 训练数据集与测试数据集
    BraTS_train_root = "E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    BraTS_test_root = "E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    train_loader = get_dataloader(slice_size=slice_size, root_dir=BraTS_train_root, batch_size=1, shuffle=True, num_workers=2, mask_rate=1, binary_mask=train_binary_mask, mode='train')
    test_loader = get_dataloader(slice_size=slice_size, root_dir=BraTS_test_root, batch_size=1, shuffle=False, num_workers=4, mask_rate=1, binary_mask=test_binary_mask, mode='eval')

    # 初始化用于跟踪最佳模型的变量
    best_loss = float('inf')
    # 训练网络
    print("------------------------------------------------------")
    print("Start Training")
    logging.info("------------------------------------------------------")
    logging.info("Training Start")
    for epoch in range(epochs):
        # 确保模型处于训练模式
        net.train()
        running_loss = 0.0
        slice_total = slice_size * train_loader.__len__()
        i = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch-person") as pbar:
            for masked_images, original_images in pbar:

                # 将数据和标签移动到设备上
                masked_images = masked_images.to(device)
                original_images = original_images.to(device)

                # 交换维度Batch_size和slice_size
                # 将slice_size作为真实的Batch_size
                # Batch_size设置为1，交换后代表单通道图像
                masked_images = swap_batch_slice_dimensions(masked_images)
                original_images = swap_batch_slice_dimensions(original_images)

                # 清空之前的梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = net(masked_images)

                # 计算损失
                loss_value = loss.calculate_loss(outputs, original_images, binary_masks=train_binary_mask)

                # 反向传播
                loss_value.backward()

                # 更新模型参数
                optimizer.step()

                # 累计损失并显示批次损失均值
                running_loss += loss_value.item()
                # 更新进度条描述
                current_slice = i + 1 + epoch * slice_size
                avg_loss = running_loss / (i + 1)  # 计算当前平均损失
                pbar.set_description(f"Epoch {epoch + 1}/{epochs} Slice {current_slice}/{slice_total}")
                pbar.set_postfix(loss=avg_loss)  # 显示当前批次的平均损失
                pbar.update()
                i += 1
            # 记录训练损失到 TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            # 调整学习率
            scheduler.step()

        # 验证模型性能
        net.eval()  # 设置模型为评估模式
        test_loss = 0.0
        avg_psnr = [0.0] * 4
        avg_ssim = [0.0] * 4
        count = 0

        start_time = time.time()  # 开始计时

        with torch.no_grad():  # 关闭梯度计算
            for masked_images, original_images in test_loader:
                masked_images = masked_images.to(device)
                original_images = original_images.to(device)
                # 交换维度Batch_size和slice_size
                # 将slice_size作为真实的Batch_size
                # Batch_size设置为1，交换后代表单通道图像
                masked_images = swap_batch_slice_dimensions(masked_images)
                original_images = swap_batch_slice_dimensions(original_images)

                outputs = net(masked_images)

                # 计算损失
                test_loss += loss.calculate_loss(outputs, original_images, binary_masks=test_binary_mask).item()
                # 计算 PSNR 和 SSIM
                current_psnr, current_ssim = eval(outputs, original_images, binary_masks=test_binary_mask)
                # 累加每个象限的 PSNR 和 SSIM
                for j in range(4):
                    avg_psnr[j] += current_psnr[j]
                    avg_ssim[j] += current_ssim[j]
                count += 1

        test_loss /= len(test_loader)
        avg_psnr_total = [x / count for x in avg_psnr]
        avg_ssim_total = [x / count for x in avg_ssim]

        end_time = time.time()  # 结束计时
        total_time = end_time - start_time  # 计算总时间

        print(f"Total evaluation time: {total_time:.2f} seconds")
        # 打印结果
        print(f"验证损失: {test_loss:.4f}")
        # print("平均 PSNR: ", " ".join([f"{x:.4f}" for x in avg_psnr_total]))
        # print("平均 SSIM: ", " ".join([f"{x:.4f}" for x in avg_ssim_total]))

        # 打印每种模态的详细 PSNR 和 SSIM
        print(f"验证 PSNR T1c {avg_psnr_total[0]:.4f}, T1n {avg_psnr_total[1]:.4f}, T2w {avg_psnr_total[2]:.4f}, T2f {avg_psnr_total[3]:.4f}")
        print(f"验证 SSIM T1c {avg_ssim_total[0]:.4f}, T1n {avg_ssim_total[1]:.4f}, T2w {avg_ssim_total[2]:.4f}, T2f {avg_ssim_total[3]:.4f}")

        # 记录验证损失和指标到 TensorBoard
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('PSNR/T1c', avg_psnr_total[0], epoch)
        writer.add_scalar('PSNR/T1n', avg_psnr_total[1], epoch)
        writer.add_scalar('PSNR/T2w', avg_psnr_total[2], epoch)
        writer.add_scalar('PSNR/T2f', avg_psnr_total[3], epoch)
        writer.add_scalar('SSIM/T1c', avg_ssim_total[0], epoch)
        writer.add_scalar('SSIM/T1n', avg_ssim_total[1], epoch)
        writer.add_scalar('SSIM/T2w', avg_ssim_total[2], epoch)
        writer.add_scalar('SSIM/T2f', avg_ssim_total[3], epoch)

        # 在每个epoch后记录训练和验证结果
        logging.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}, Validation Loss: {test_loss:.4f}")
        logging.info(f"Validation PSNR: {' '.join([f'{x:.4f}' for x in avg_psnr_total])}")
        logging.info(f"Validation SSIM: {' '.join([f'{x:.4f}' for x in avg_ssim_total])}")
        logging.info("------------------------------------------------------")

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            best_model_path = os.path.join(directory, f'best_model_epoch_{epoch + 1}.ckpt')
            torch.save(net.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch + 1} to {best_model_path}")

        # 保存定期的检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(directory, f'checkpoint_epoch_{epoch + 1}.ckpt')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")
    # args = get_args()
     # 关闭 TensorBoard 记录器
    writer.close()