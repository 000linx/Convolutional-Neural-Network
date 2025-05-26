from types import new_class
import torch
import torch.nn as nn
import torch.optim as optim # 优化器
from torch.utils.data import DataLoader # 数据加载器
from torchvision import datasets, transforms # 数据集和数据预处理
from tqdm import tqdm # 进度条
import os
from cnn import simplecnn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对图像做变换
train_transform = transforms.Compose([
    transforms.Resize([224,224]), # 将图片大小调整为224*224
    transforms.ToTensor(), # 将图片转换为张量
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 将图片归一化到[-1,1]
])

test_transform = transforms.Compose([
    transforms.Resize([224,224]), # 将图片大小调整为224*224
    transforms.ToTensor(), # 将图片转换为张量
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 将图片归一化到[-1,1]
])

# 定义数据集加载类
trainset = datasets.ImageFolder(os.path.join(r"CNN\TestProject\Datasets\COVID_19_Radiography_Dataset","train"), # 拼接路径，找到训练集
                                transform = train_transform)

testset = datasets.ImageFolder(os.path.join(r"CNN\TestProject\Datasets\COVID_19_Radiography_Dataset","test"),
                               transform=test_transform)

# 定义数据加载器
train_loader = DataLoader(trainset, batch_size=32, num_workers=0, shuffle=True) # 训练集,参数解释： 32个样本即一次传入32张图片, 0个进程()即关闭多线程, 打乱数据

test_loader = DataLoader(testset, batch_size=32, num_workers=0, shuffle=False)


def train(model, train_loader, criterion, optimizer, num_epochs): # 定义训练函数，参数解释：模型，训练集，损失函数，优化器，训练轮数
    best_acc = 0.0
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader , desc=f"epoch:{epoch + 1} / {num_epochs}", unit="batch"): # 进度条，显示训练进度
            inputs, labels = inputs.to(device), labels.to(device) # 将数据传入GPU
            optimizer.zero_grad() # 清空梯度
            outputs = model(inputs) # 前向传播
            loss  = criterion(outputs, labels)
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            running_loss += loss.item() * inputs.size(0) # 计算损失, 乘以inputs.size(0)是因为损失是平均的，乘以inputs.size(0)可以得到总的损失
    epoch_loss = running_loss / len(train_loader.dataset) # 总损失/数据集的大小 为我们每轮的损失
    print(f"epoch[{epoch + 1} / {num_epochs}, Train_loss:{epoch_loss:.4f}]")

    accuracy = evaluate(model, test_loader,criterion)
    if accuracy > best_acc:
        best_acc = accuracy
        save_model(model, save_path)
        print("model saved with best accuracy", best_acc)

# 定义测试函数，参数解释：模型，测试集
def evaluate(model, test_loader, criterion):
    model.eval() #  指定模型为验证模式
    test_loss = 0.0 # 测试的初始化loss
    correct = 0
    total = 0
    # 关闭梯度计算，因为在测试时不需要计算梯度，也不需要更新参数
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * input.size(0)
            _,predicted = torch.max(outputs , 1) # 找到概率最大的下标, 即预测的类别
            total += labels.size(0) # 计算总样本数
            correct += (predicted == labels).sum().item() # 计算正确的样本数，即预测的类别和真实的类别相同的样本数
    avg_loss = test_loss / len(test_loader.dataset) # 计算平均损失
    accuracy = 100.0 * correct / total # 计算准确率
    print(f"Test_loss:{avg_loss:.4f}, Accuracy:{accuracy:.2f}%") # 输出平均损失和准确率
    return accuracy
    
# 保存
def save_model(model, save_path):
    torch.save(model.state_dict(),save_path) # 保存模型的参数


if __name__ == "__main__":
    num_echops = 10
    learing_rate = 0.001
    num_class = 4
    save_path = r"model_pth\best.pth"
    model = simplecnn(num_class).to(device) # 实例化模型
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr = learing_rate) # 优化器，Adam优化器，学习率为0.001
    train(model, train_loader, criterion, optimizer, num_echops) # 使用训练集进行训练
    evaluate(model, test_loader, criterion) # 使用测试集进行测试