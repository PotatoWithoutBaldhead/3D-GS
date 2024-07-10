import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 定义特征提取模型
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.features.children())[:])

    def forward(self, x):
        x = self.features(x)
        return x

vgg16 = models.vgg16(pretrained=True)
feature_extractor = FeatureExtractor(vgg16).eval()

# 加载和预处理图像
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

image1 = load_image('path_to_image1.jpg')
image2 = load_image('path_to_image2.jpg')

# 使一幅图像的像素需要梯度
image1.requires_grad = True

# 定义优化器
optimizer = optim.SGD([image1], lr=0.1)

# 前向传播
features1 = feature_extractor(image1)
features2 = feature_extractor(image2)

# 计算输出差异
loss = nn.MSELoss()(features1, features2)

# 反向传播
loss.backward()

# 更新图像像素
optimizer.step()

# 重置梯度
optimizer.zero_grad()

# 将更新后的图像保存到磁盘
updated_image = image1.detach().squeeze().permute(1, 2, 0).numpy()
updated_image = (updated_image - updated_image.min()) / (updated_image.max() - updated_image.min())
updated_image = (updated_image * 255).astype(np.uint8)
Image.fromarray(updated_image).save('updated_image1.jpg')
