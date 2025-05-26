import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# 数据集类
class AnimeFaceDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# 加载图片文件路径
file_list = glob.glob(r"CNN\Ani_face\ani_face\*.jpg")
dataset = AnimeFaceDataset(file_list, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 128),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 3, kernel_size=5, padding=2),
            nn.Tanh()  # 输出范围为 [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# GAN 模型
class GAN:
    def __init__(self, discriminator, generator, latent_dim, device, num_img=10):
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.device = device
        self.num_img = num_img  # 添加 num_img 属性
        self.discriminator.to(device)
        self.generator.to(device)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            i=0
            print(len(dataloader))
            for real_images in dataloader:
                print(i)
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # 训练判别器
                self.d_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # 真实图像的损失
                real_outputs = self.discriminator(real_images)
                d_loss_real = self.loss_fn(real_outputs, real_labels)

                # 生成假图像
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                fake_outputs = self.discriminator(fake_images.detach())  # 不更新生成器的梯度
                d_loss_fake = self.loss_fn(fake_outputs, fake_labels)

                # 判别器总损失
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # 训练生成器
                self.g_optimizer.zero_grad()
                fake_outputs = self.discriminator(fake_images)
                g_loss = self.loss_fn(fake_outputs, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                i+=1

            print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            # 保存生成的图像
            if not os.path.exists("gen_ani"):
                os.makedirs("gen_ani")
            with torch.no_grad():
                noise = torch.randn(self.num_img, self.latent_dim).to(self.device)
                generated_images = self.generator(noise)
                save_image(generated_images, f"gen_ani/generated_img_{epoch:03d}.png", normalize=True)

# 初始化模型
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
discriminator = Discriminator()
generator = Generator(latent_dim)
gan = GAN(discriminator, generator, latent_dim, device, num_img=16)  # 添加

# 编译模型
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
loss_fn = nn.BCELoss()
gan.compile(d_optimizer, g_optimizer, loss_fn)

# 训练模型
epochs = 4
gan.train(dataloader, epochs)