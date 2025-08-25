#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    total = 0  # 用于记录总样本数

    # 创建数据加载器
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)

    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)

            # 前向传播计算预测
            log_probs = net_g(data)

            # 累加批次损失
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # 计算预测结果
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).sum().item()  # 正确预测的数量

            total += target.size(0)  # 累加总样本数

    # 计算平均损失
    test_loss /= total

    # 计算精度
    accuracy = 100.00 * correct / total

    return accuracy, test_loss



