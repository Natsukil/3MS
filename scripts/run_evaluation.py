from dataset_conversion.BraTsData_person import Dataset_brats
from torch.utils.data import DataLoader
import torch
from networks.UNet import UNet
from networks.S_UNet import S_UNet
from utils.swap_dimensions import swap_batch_slice_dimensions
from training.loss_functions import LossFunctions
from eval import evaluate_model
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluation(load_dir, show_image=False):
    # 训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating on:", device)

    # 定义网络模型
    net = UNet(in_channels=1, out_channels=1)
    # net = S_UNet(in_channels=1, base_filters=32, bias=False)
    net.to(device)

    net.load_state_dict(torch.load(load_dir, map_location=device))
    print(f"load model from {load_dir}")

    test_binary_mask = '1000'

    root_dir = "E:\Work\dataset\ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    dataset = Dataset_brats(root_dir=root_dir, slice_deep=8, binary_mask=test_binary_mask, mask_rate=1, mode='eval')
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 定义损失函数
    loss = LossFunctions()

    # 定义性能评价指标
    eval = evaluate_model

    net.eval()  # 设置模型为评估模式
    test_loss = 0.0
    avg_psnr = [0.0] * 4
    avg_ssim = [0.0] * 4
    count = 0
    with torch.no_grad():  # 关闭梯度计算
        for masked_images, original_images in tqdm(test_loader, desc="Evaluating", unit="batch-person"):
            masked_images = masked_images.to(device)
            original_images = original_images.to(device)
            # 交换维度Batch_size和Slice_num
            # 将Slice_num作为真实的Batch_size
            # Batch_size设置为1，交换后代表单通道图像
            masked_images = swap_batch_slice_dimensions(masked_images)
            original_images = swap_batch_slice_dimensions(original_images)

            outputs = net(masked_images)
            print(outputs.shape)
            if show_image:
                original_images = original_images.to('cpu')
                outputs = outputs.to('cpu')
                plt.figure(figsize=(10, 5))

                # 显示 original_images
                plt.subplot(1, 2, 1)  # 1行3列的第1个位置
                plt.imshow(original_images[1, 0, :, :], cmap='gray')
                plt.colorbar()  # 添加颜色条
                plt.title('original_images')  # 添加标题
                plt.axis('off')  # 关闭坐标轴显示

                # 显示 masked_result
                plt.subplot(1, 2, 2)  # 1行3列的第2个位置
                plt.imshow(outputs[1, 0, :, :], cmap='gray')
                plt.colorbar()  # 添加颜色条
                plt.title('Masked Result')
                plt.axis('off')  # 关闭坐标轴显示
                plt.show()
                break

            # 计算损失
            test_loss += loss.calculate_loss(outputs, original_images, binary_masks=test_binary_mask).item()
            # 计算 PSNR 和 SSIM
            current_psnr, current_ssim = eval(outputs, original_images)
            # 累加每个象限的 PSNR 和 SSIM
            for j in range(4):
                avg_psnr[j] += current_psnr[j]
                avg_ssim[j] += current_ssim[j]
            count += 1

    if show_image:
        exit()

    test_loss /= len(test_loader)
    avg_psnr_total = [x / count for x in avg_psnr]
    avg_ssim_total = [x / count for x in avg_ssim]
    # 打印结果
    print(f"验证损失: {test_loss:.4f}")
    # print("平均 PSNR: ", " ".join([f"{x:.4f}" for x in avg_psnr_total]))
    # print("平均 SSIM: ", " ".join([f"{x:.4f}" for x in avg_ssim_total]))

    # 打印每种模态的详细 PSNR 和 SSIM
    print(
        f"验证 PSNR T1n {avg_psnr_total[0]:.4f}, T1c {avg_psnr_total[1]:.4f}, T2w {avg_psnr_total[2]:.4f}, T2f {avg_psnr_total[3]:.4f}")
    print(
        f"验证 SSIM T1n {avg_ssim_total[0]:.4f}, T1c {avg_ssim_total[1]:.4f}, T2w {avg_ssim_total[2]:.4f}, T2f {avg_ssim_total[3]:.4f}")


if __name__ == '__main__':
    evaluation("result/models/UNet/1000-05-18-01-20-12/checkpoint_epoch_110.ckpt", show_image=True)
