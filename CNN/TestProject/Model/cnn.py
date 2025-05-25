import torch.nn as nn


# 定义一个简单的卷积神经网络模型
class simplecnn(nn.Module):
    def __init__(self, num_class): # num_classes 是分类的类别数量
        super().__init__() # 调用父类的初始化方法，这是必备的

        # 定义网络的特征提取部分 
        self.features = nn.Sequential(
            # 输入的通道数为3，输出的通道数为16，卷积核大小为3，步长为1，填充为1，卷积层
            nn.Conv2d(3,16,kernel_size = 3, stride = 1, padding = 1), # 保证输入输出的尺寸不变
            # 激活函数，使用ReLU，增加非线性特性
            nn.ReLU(),
            # 最大池化层，池化核大小为2，步长为2，池化层
            nn.MaxPool2d(kernel_size = 2, stride = 2), # 池化之后图像尺寸减半
            # 第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为3，步长为1，填充为1，卷积层
            nn.Conv2d(16,32, kernel_size = 3, stride = 1, padding  = 1),
            # 再使用激活函数
            nn.ReLU(),
            # 再使用最大池化层
            nn.MaxPool2d(kernel_size = 2, stride = 2) # 池化之后图像尺寸减半
        )

        # 定以全连接层，用于将特征图展平为一维向量，并进行分类
        self.classifier = nn.Sequential(
             # 输入特征图的大小为32*56*56，输出特征图的大小为128，全连接层
            nn.Linear(32*56*56, 128),
            # 再使用激活函数
            nn.ReLU(),
            # 再使用全连接层，输入特征图的大小为128，输出特征图的大小为num_classes，全连接层
            nn.Linear(128, num_class)
        )

    # 定义前向传播函数，输入x是一个批次的图像数据
    def forward(self, x):
        # 首先通过特征提取部分，将输入的图像数据转换为特征图
        x = self.features(x)
        # 再将提取出的特征图展平为一维向量，x.size(0)是批次的大小即batch，-1表示自动计算其他维度的大小
        x = x.view(x.size(0), -1)   
        # 再通过全连接层进行分类
        x = self.classifier(x) 
        return x # 返回分类结果