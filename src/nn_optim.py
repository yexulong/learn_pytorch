#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# P24 优化器（一）
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from nn_seq import LearnSeq

dataset = torchvision.datasets.CIFAR10('../cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

loss = nn.CrossEntropyLoss()
learn_seq = LearnSeq()
optim = torch.optim.SGD(learn_seq.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = learn_seq(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print('当前loss:', running_loss)
