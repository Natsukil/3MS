# 3MS(Multi-Model MRI Image Synthesis)
Multi-model (Multi-Sequence) MRI Image Synthesis Task 多模态（多序列）MRI图像生成

## 数据集 
BraTS2023 脑胶质瘤数据集
包含4个序列：T1, T1ce, T2, FLAIR
### 数据集预处理 
- [x] 使用2D图像预处理
- [x] 将4个序列的图像合拼接成成一个图像，组合成3通道480 * 480的图像为一个样本

## 思路
- 训练：
1. 对图像进行遮蔽处理，在四个序列的图像上都进行随机遮蔽
- 随机遮蔽/基于注意力的遮蔽/全部遮蔽
- 保证每个遮蔽的区块在另外三个模态上至少有一个像素值
2. 通过模型进行重建任务，评价重建图像与原始图像的误差
- 预测：
- 对特定模态的图像进行全部遮蔽（遮蔽的特殊情况），进行重建任务，以达到生成任务的目的

## 将要尝试的模型
1. U-Net
2. nnU-Net
3. U-Net++
4. GAN
5. Diffusion Model
6. VAE
