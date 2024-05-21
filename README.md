# 3MS(Multi-Model MRI Image Synthesis)
Multi-model (Multi-Sequence) MRI Image Synthesis Task 多模态（多序列）MRI图像生成

## 数据集 
BraTS2023 脑胶质瘤数据集
包含4个序列：T1, T1ce, T2, FLAIR
### 数据集预处理 
- [x] 使用2D图像预处理
- [X] 对图像进行剪切，将宽高修改为192*192
- [X] 对每个序列进行裁剪，根据参数Slice_size控制，选择中心部分的图像
- [x] 将4个序列的图像合拼接成成一个图像，组合成单通道384 * 384的图像为一个样本
### 遮蔽算法
参数：遮蔽比率，遮蔽编码（4位），遮蔽块大小
- 根据遮蔽编码，在要执行遮蔽的图像上进行遮蔽
- 将图像划分为规则的遮蔽块，根据遮蔽比率计算要遮蔽的块数
- 选择遮蔽块时，至少保证被遮蔽的区域至少在一个序列上有数据
- 最大遮蔽率：75%（保证每块被遮蔽的数量相同）
## 思路
- 训练：
1. 对图像进行遮蔽处理，在四个序列的图像上进行遮蔽
- 随机遮蔽/基于注意力的遮蔽/全部遮蔽
- 保证每个遮蔽的区块在另外三个模态上至少有一个像素值
- 对一至三个序列全部遮蔽
2. 通过模型进行重建任务，评价重建图像与原始图像的误差
- 预测：
- 对特定模态的图像进行全部遮蔽（遮蔽的特殊情况），进行重建任务，以达到生成任务的目的

## 训练设置
### 将要尝试的模型
1. U-Net及其改进版本
2. nnU-Net
3. U-Net++
4. GAN
5. Diffusion Model
6. VAE
### 损失函数
重建损失：计算每个序列的MSE并相加
- 调整思路
根据不同的输入序列组合，设置MSE的分配权重：是否可以设置为可以学习的参数？
后续：
- 调整权重
- L1范数
- 感知损失LPIPS
- 重建损失
### 学习率
- UNet:2e-4
### 评价指标
- PSNR（基于MSE）
- SSIM

后续
- MS-SSIM
- LPIPS

### Lr_scheduler
torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

## Train
- 使用完整的参数
```angular2html
python scripts/run_training.py --config config/UNet_2d.yaml --device cuda:0 --model UNet --pretrain True --load_dir best.ckpt
```
- 使用config设置参数
```angular2html
python scripts/run_training.py
```
获取帮助
```angular2html
python scripts/run_training.py -h
```
