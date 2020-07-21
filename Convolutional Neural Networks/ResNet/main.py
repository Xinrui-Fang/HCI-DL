#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from ResNet import ResNet18
from torch.utils.tensorboard import SummaryWriter

modelPath = './model.pkl'

# 定义Summary_Writer
writer = SummaryWriter('./Result')   # 数据存放在这个文件夹


# In[ ]:


def train(args, model, device, train_loader, optimizer, epoch):
    model.train() # 必备，将模型设置为训练模式
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    for batch_idx, (data, target) in enumerate(train_loader): # 从数据加载器迭代一个batch的数据
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 清除所有优化的梯度
        output = model(data) # 喂入数据并前向传播获取输出
        loss = criterion(output, target).to(device) # 调用损失函数计算损失
        loss.backward() # 反向传播
        optimizer.step() #更新参数

        if batch_idx % args.log_interval == 0: # 根据设置的显示间隔输出训练日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    writer.add_scalar('train_loss', loss.item(), epoch)

def test(args, model, device, test_loader, epoch):
    model.eval() # 必备，将模型设置为评估模式
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    with torch.no_grad(): # 禁用梯度计算
        for data, target in test_loader: # 从数据加载器迭代一个batch的数据
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).to(device).item() # # sum up batch loss
            _, predicted = torch.max(output.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('accuracy', correct / len(test_loader.dataset), epoch)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='B',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_known_args()[0]
    use_cuda = not args.no_cuda and torch.cuda.is_available() # 根据输入参数和实际cuda的有无决定是否使用GPU

    torch.manual_seed(args.seed) # 设置随机种子，保证可重复性

    device = torch.device("cuda" if use_cuda else "cpu") # 设置使用CPU or GPU

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} # 设置数据加载的子进程数；是否返回之前将张量复制到cuda的页锁定内存
    

    trainset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)


    model = ResNet18().to(device) # 实例化网络模型
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # 实例化求解器

    for epoch in range(1, args.epochs + 1): # 循环调用train() and test() 进行epoch迭代
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

        if epoch % 10 == 0: # save model every 10 epoches
            torch.save(model, './model.pkl')

main()

