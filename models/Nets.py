#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        return x



class CNNFmnist(nn.Module):
    def __init__(self, args):
        super(CNNFmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HARCNN(nn.Module):
    """
    使用1D卷积处理UCI HAR数据:
      输入: (N, 1, 561)
      输出: (N, 6)   # 6分类
    """

    def __init__(self, args):
        super(HARCNN, self).__init__()
        # 第1段：Conv1d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 第2段：Conv1d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 根据卷积+池化后的输出形状确定线性层输入维度:
        #  - conv1: 输入 561 -> 输出 559
        #  - pool1: 559 -> 约 279
        #  - conv2: 279 -> 输出 277
        #  - pool2: 277 -> 约 138
        # 故最终特征长度 = 32 * 138 = 4416
        self.fc1 = nn.Linear(32 * 138, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        """
        x: (batch_size, 1, 561)
        如果数据是 (batch_size, 561)，可先 x = x.unsqueeze(1) 转成 (batch_size, 1, 561)
        """
        x = x.view(x.size(0), 1, -1)  # => (64, 1, 561)
        print("Input shape:", x.shape)
        # 卷积1
        x = self.conv1(x)  # => (N,16,559)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # => (N,16,279)

        # 卷积2
        x = self.conv2(x)  # => (N,32,277)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # => (N,32,138)

        # 拉平
        x = x.view(x.size(0), -1)  # => (N, 32*138=4416)

        # 全连接
        x = F.relu(self.fc1(x))
        out = self.fc2(x)  # => (N,6)
        return out

class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()

        # 加载预训练的 ResNet18
        self.model = models.resnet18(pretrained=True)

        # 修改最后的全连接层
        self.model.fc = nn.Linear(self.model.fc.in_features, args.num_classes)

    def forward(self, x):
        # 向前传播
        return self.model(x)
