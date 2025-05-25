import torch.nn as nn


class FaceCNN(nn.Module):
    def __init__(self):
        super.__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32x3 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x64 -> 16x16x64 
        )