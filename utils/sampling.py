#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from utils.options import args_parser
from collections import defaultdict
import matplotlib.pyplot as plt

def create_non_iid_data_indices(dataset, num_users, alpha=0.9):
    """
        Sample non-IID client data from MNIST dataset using Dirichlet distribution
        :param dataset: MNIST dataset
        :param num_users: Number of users
        :param alpha: Dirichlet distribution parameter
        :return: dict of image index
        """
    targets = dataset.targets.numpy()
    num_classes = len(np.unique(targets))

    # 创建一个字典存储每个参与方的数据索引
    dict_users = defaultdict(list)

    # 按类别分割数据索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # 生成 Dirichlet 分布
    class_sizes = [len(class_indices[c]) for c in range(num_classes)]
    class_participant_indices = []
    for c in range(num_classes):
        class_indices_list = np.random.permutation(class_indices[c])
        splits = np.random.dirichlet([alpha] * num_users)
        class_participant_indices.append(
            np.split(class_indices_list, (np.cumsum(splits) * class_sizes[c]).astype(int)[:-1]))

    # 分配数据索引给参与方
    for c in range(num_classes):
        for participant_id in range(num_users):
            dict_users[participant_id].extend(class_participant_indices[c][participant_id])

    return dict_users

def create_non_iid_data_indices_cifar(dataset, num_users, alpha=1):
    """
        Sample non-IID client data from MNIST dataset using Dirichlet distribution
        :param dataset: MNIST dataset
        :param num_users: Number of users
        :param alpha: Dirichlet distribution parameter
        :return: dict of image index
        """
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    # 创建一个字典存储每个参与方的数据索引
    dict_users = defaultdict(list)

    # 按类别分割数据索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # 生成 Dirichlet 分布
    class_sizes = [len(class_indices[c]) for c in range(num_classes)]
    class_participant_indices = []
    for c in range(num_classes):
        class_indices_list = np.random.permutation(class_indices[c])
        splits = np.random.dirichlet([alpha] * num_users)
        class_participant_indices.append(
            np.split(class_indices_list, (np.cumsum(splits) * class_sizes[c]).astype(int)[:-1]))

    # 分配数据索引给参与方
    for c in range(num_classes):
        for participant_id in range(num_users):
            dict_users[participant_id].extend(class_participant_indices[c][participant_id])

    return dict_users

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    args = args_parser()


    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    # Sample non-I.I.D client data from MNIST dataset
    # :param dataset:
    # :param num_users:
    # :return:
    # """
    targets = dataset.targets.numpy()
    num_classes = len(np.unique(targets))
    alpha = 1
    dict_users = create_non_iid_data_indices(dataset, num_users, alpha)
    # plt.figure(figsize=(15, num_users * 3))
    for participant_id, indices in dict_users.items():
        label_counts = np.zeros(num_classes, dtype=int)
        for idx in indices:
            label_counts[targets[idx]] += 1
        print(f"Participant {participant_id}:")
        for label in range(num_classes):
            print(f"  Label {label}: {label_counts[label]} samples")
            # Plot label distribution
    #     plt.subplot(num_users, 1, participant_id + 1)
    #     plt.bar(range(num_classes), label_counts, tick_label=[str(i) for i in range(num_classes)])
    #     plt.xlabel('Labels')
    #     plt.ylabel('Number of samples')
    #     plt.title(f'Participant {participant_id}')
    #     plt.ylim(0, max(label_counts) + 10)
    #     plt.xticks(fontsize=8)
    #     plt.yticks(fontsize=8)
    #
    # plt.tight_layout()
    # plt.savefig('./save/Dis.png')
    # #这里的noniid是客户端拿到的标签不一样实现的
    # num_shards, num_imgs = 200, 300
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    #
    # #print(idx_shard)
    # #print(dict_users)
    # #print(idxs)
    # #print(labels)
    #
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # #以标签为升序排序，且与对应的索引匹配
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # #idx保存第一行的索引
    # idxs = idxs_labels[0,:]
    #
    # #print(idxs)
    #
    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     #print(rand_set)
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # print(dict_users)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    alpha = 0.9
    dict_users = create_non_iid_data_indices_cifar(dataset, num_users, alpha)
    # plt.figure(figsize=(15, num_users * 3))
    for participant_id, indices in dict_users.items():
        label_counts = np.zeros(num_classes, dtype=int)
        for idx in indices:
            label_counts[targets[idx]] += 1
        print(f"Participant {participant_id}:")
        for label in range(num_classes):
            print(f"  Label {label}: {label_counts[label]} samples")
            # Plot label distribution
    #     plt.subplot(num_users, 1, participant_id + 1)
    #     plt.bar(range(num_classes), label_counts, tick_label=[str(i) for i in range(num_classes)])
    #     plt.xlabel('Labels')
    #     plt.ylabel('Number of samples')
    #     plt.title(f'Participant {participant_id}')
    #     plt.ylim(0, max(label_counts) + 10)
    #     plt.xticks(fontsize=8)
    #     plt.yticks(fontsize=8)
    #
    # plt.tight_layout()
    # plt.savefig('./save/Dis.png')
    return dict_users

def fmnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from FMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def fmnist_noniid(dataset, num_users):
    targets = dataset.targets.numpy()
    num_classes = len(np.unique(targets))
    alpha = 1
    dict_users = create_non_iid_data_indices(dataset, num_users, alpha)
    # plt.figure(figsize=(15, num_users * 3))
    for participant_id, indices in dict_users.items():
        label_counts = np.zeros(num_classes, dtype=int)
        for idx in indices:
            label_counts[targets[idx]] += 1
        print(f"Participant {participant_id}:")
        for label in range(num_classes):
            print(f"  Label {label}: {label_counts[label]} samples")
            # Plot label distribution
    #     plt.subplot(num_users, 1, participant_id + 1)
    #     plt.bar(range(num_classes), label_counts, tick_label=[str(i) for i in range(num_classes)])
    #     plt.xlabel('Labels')
    #     plt.ylabel('Number of samples')
    #     plt.title(f'Participant {participant_id}')
    #     plt.ylim(0, max(label_counts) + 10)
    #     plt.xticks(fontsize=8)
    #     plt.yticks(fontsize=8)
    #
    # plt.tight_layout()
    # plt.savefig('./save/Dis.png')
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
