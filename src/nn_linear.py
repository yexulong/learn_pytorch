import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('../cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class LearnLinear(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input_data):
        return self.linear1(input_data)


learn_linear = LearnLinear()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = learn_linear(output)
    print(output.shape)
