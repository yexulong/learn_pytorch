"""减少参数

"""

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('../cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class LearnMaxpool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input_data):
        output_data = self.maxpool1(input_data)
        return output_data


learn_maxpool = LearnMaxpool()

writer = SummaryWriter('../logs_maxpool')
for index, data in enumerate(dataloader):
    imgs, targets = data
    output = learn_maxpool(imgs)
    writer.add_images('input1', imgs, index)
    # output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output1', output, index)
writer.close()
