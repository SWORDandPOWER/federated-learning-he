#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
from sympy import together

print("***************FLHE is Running****************", flush=True)

import os, sys,io

# 解决powershell不能实时输出
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 解决powershell实时输出乱码
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import random
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fmnist_iid, fmnist_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResNet18, CNNFmnist
from models.test import test_img
from roles.TA import TA
import tenseal as ts

torch.manual_seed(43)
random.seed(43)
np.random.seed(43)
if torch.cuda.is_available():
    torch.cuda.manual_seed(43)
    torch.cuda.manual_seed_all(43)

if __name__ == '__main__':
    args = args_parser()

    if args.security == 'ckks':
        print("---------CKKS密钥生成中--------")
    elif args.security == 'paillier':
        print("---------Paillier密钥生成中--------")

    st = time.time()
    ta = TA()
    ed = time.time()
    print(f"Key Gen {(ed - st) * 1000} ms")


    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )

    # ---------- 数据集加载 ----------
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform_train)
        dataset_test  = datasets.MNIST('./data/mnist/', train=False, download=True, transform=transform_test)
        dict_users = mnist_iid(dataset_train, args.num_users) if args.iid else mnist_noniid(dataset_train,args.num_users)
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform_train)
        dataset_test  = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform_test)
        dict_users = cifar_iid(dataset_train, args.num_users) if args.iid else cifar_noniid(dataset_train,args.num_users)
    elif args.dataset == 'fmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.FashionMNIST('./data/fmnist', train=True, download=True, transform=transform_train)
        dataset_test  = datasets.FashionMNIST('./data/fmnist', train=False, download=True, transform=transform_test)
        dict_users = fmnist_iid(dataset_train, args.num_users) if args.iid else fmnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # ---------- 建模型 ----------
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = CNNFmnist(args=args).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = ResNet18(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # ---------- 【改动1】统一设备 ----------
    device = next(net_glob.parameters()).device   # 当前设备
    w_glob = net_glob.state_dict()
    for k in w_glob:
        w_glob[k] = w_glob[k].to(device)          # 搬到同一设备
    g_glob = {}

    # ---------- 训练 ----------
    loss_train, test_acc = [], []
    train_time_round,ency_time_round = [], []   # 每一次通讯轮次的本地训练耗时总和，加密耗时总和
    together_time_round,dec_time_round = [], []  # 每一次通讯轮次的聚合耗时，解密耗时
    time_count = {'train_time_per_client': 0, 'ency_time_per_client': 0}

    for iter in range(args.epochs):
        loss_locals, w_locals, g_locals = [], [], []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # 修改后的第一行：输出轮次 + 选中客户端
        print(f"------------ Round {iter + 1}: Selected clients: {idxs_users.tolist()}------------")

        print("------------ 本轮次所有选中客户端本地训练中 ------------")
        for idx in idxs_users:
            print(f"------ Client {idx}:  开始本地训练")  # 确认每个客户端
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            st = time.time()
            w, loss = local.train(net=copy.deepcopy(net_glob).to(device))
            ed = time.time()
            print(f"Client {idx}: Train Time {(ed - st) * 1000} ms")  # 输出本轮该客户端本地训练耗时
            time_count['train_time_per_client'] += (ed - st) * 1000  # 累加该客户端的训练耗时

            # ---------- 不再重新 w_glob = ... ----------
            g = {}
            for name in w.keys():
                g[name] = w[name].to(device) - w_glob[name].to(device)  # 同一设备

            # ---------- 加密前搬到 CPU ----------
            local_model_updates = torch.cat(
                [param.detach().view(-1) for param in g.values()], dim=0
            ).cpu().numpy()

            ctx = ts.context_from(ta.ckks_public_key)
            if args.security == 'ckks':
                print(f"------ Client {idx}: CKKS加密本地梯度中")  # 客户端ID
                st = time.time()
                enc_local_model_updates = ts.ckks_vector(ctx, local_model_updates)
                ed = time.time()
                print(f"Client {idx}: Encrypt Time {(ed - st) * 1000} ms") # 输出本轮该客户端加密耗时
                time_count['ency_time_per_client'] += (ed - st) * 1000  #累加该客户端的加密耗时
                g_locals.append(enc_local_model_updates)
            elif args.security == 'paillier':
                print(f"------ Client {idx}: Paillier加密本地梯度中")  # 加客户端ID
                st = time.time()
                enc_local_model_updates = np.vectorize(
                    lambda x: ta.paillier_public_key.encrypt(int(x * 1e5)), otypes=[object]
                )(local_model_updates)
                ed = time.time()
                time_count['ency_time_per_client'] += (ed - st) * 1000  #累加该客户端的加密耗时
                g_locals.append(enc_local_model_updates)
            elif args.security == 'no':
                g_locals.append(local_model_updates)
            loss_locals.append(copy.deepcopy(loss))

        # ---------- FedAvg ----------
        print("-----------FedAvg聚合本地模型中------------")
        st = time.time()
        g_glob_fl = g_locals[0]
        for i in range(1, len(g_locals)):
            g_glob_fl = g_glob_fl + g_locals[i]
        ed = time.time()
        together_time = (ed-st) * 1000  # 本轮聚合耗时
        together_time = round(together_time, 2) # 保留2位小数
        together_time_round.append(together_time)  # 本轮聚合耗时添加到列表里

        dec_time = -1 # 默认解密时间为-1

        if args.security == 'ckks':
            ctx = ts.context_from(ta.ckks_secret_key)
            st = time.time()
            print("----------CKKS解密全局模型-------------")
            g_glob_fl = g_glob_fl.decrypt(ctx.secret_key())
            g_glob_fl = np.array(g_glob_fl, dtype=np.float32)
            ed = time.time()
            dec_time  = (ed - st) * 1000  # 本轮解密耗时
            dec_time = round(dec_time, 2)

        elif args.security == 'paillier':
            st = time.time()
            print("----------Paillier解密全局模型-------------")
            g_glob_fl = np.vectorize(lambda x: ta.paillier_secret_key.decrypt(x) / 1e5)(g_glob_fl)
            g_glob_fl = np.array(g_glob_fl, dtype=np.float32)
            ed = time.time()
            dec_time = (ed - st) * 1000
            dec_time = round(dec_time, 2)

        dec_time_round.append(dec_time)  # 记录每轮解密时间

        # ----------解密后搬回同一设备 ----------
        g_glob = {}
        start = 0
        for name in w_glob:
            length = w_glob[name].numel()
            g_glob[name] = torch.from_numpy(g_glob_fl[start:start + length]) \
                                 .view(w_glob[name].shape) \
                                 .to(device)           # 搬回 GPU/CPU
            start += length

        for name in w_glob:
            g_glob[name] = torch.div(g_glob[name], len(g_locals))
            w_glob[name] = w_glob[name] + g_glob[name]

        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}'.format(iter + 1))
        loss_train.append(loss_avg)

        print("-------------测试全局模型性能--------------")
        acc_test, loss_test_f = test_img(net_glob, dataset_test, args)
        print("Testing loss: {:.2f}".format(loss_test_f))
        print("Testing accuracy: {:.2f}".format(acc_test))
        test_acc.append(acc_test)

        print("本轮本地训练总耗时为：{:.2f} ms".format(round(time_count['train_time_per_client'], 2)))
        train_time_round.append(round(time_count['train_time_per_client'], 2))

        print("本轮加密总耗时为：{:.2f} ms".format(round(time_count['ency_time_per_client'], 2)))
        ency_time_round.append(round(time_count['ency_time_per_client'], 2))

        print("本轮聚合耗时为:{} ms".format(together_time))
        print("本轮解密耗时为:{} ms".format(dec_time))

        time_count['train_time_per_client']= 0
        time_count['ency_time_per_client'] = 0

    rounds = list(range(1, args.epochs + 1))
    test_acc = np.array(test_acc)
    train_time_round = np.array(train_time_round)
    ency_time_round = np.array(ency_time_round)
    together_time_round = np.array(together_time_round)
    dec_time_round = np.array(dec_time_round)
    data = {'rounds': rounds, 'accuracy': test_acc, 'train_time': train_time_round, 'ency_time': ency_time_round,
            'FedAvg': together_time_round,'dec_time': dec_time_round}
    df_data = pd.DataFrame(data)

    iid_tag = 'iid' if args.iid else 'non-iid'
    df_data.to_csv('./roles/ACC_{}_{}_{}_{}.csv'.format(args.model, args.dataset, args.security, iid_tag))