import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

print(torchvision.__version__)

vgg = models.vgg19(Pretrained=True).features
vgg.eval()

# 将图像转换为合适的格式
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])