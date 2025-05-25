import os 
import torch
from cnn import simplecnn

# 输入的图像尺寸为224*224，通道数为3，批次大小为32
x = torch.randn(32,3,224,224) 

# 实例化模型，假设我们有10个类别
model = simplecnn(num_class = 10)
output = model(x)
print(output.shape) # 输出结果为torch.Size([32, 10])，即每个样本的分类结果为10个类别