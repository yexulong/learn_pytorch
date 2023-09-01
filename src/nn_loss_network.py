# P23 损失函数与反向传播
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from nn_seq import LearnSeq

dataset = torchvision.datasets.CIFAR10('../cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

loss = nn.CrossEntropyLoss()
learn_seq = LearnSeq()
for data in dataloader:
    imgs, targets = data
    output = learn_seq(imgs)
    result_loss = loss(output, targets)
    result_loss.backward()
    print('ok')
