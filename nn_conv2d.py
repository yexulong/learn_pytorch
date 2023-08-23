# P18 神经网络-卷积层
import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class LearnNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


learn_nn = LearnNN()

writer = SummaryWriter('logs')
for index, data in enumerate(dataloader):
    imgs, targets = data
    output = learn_nn(imgs)
    writer.add_images('input1', imgs, index)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output1', output, index)

writer.close()
