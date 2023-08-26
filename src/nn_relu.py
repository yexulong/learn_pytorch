import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input_data = torch.tensor([[1, -0.5],
                           [-1, 3]])

dataset = torchvision.datasets.CIFAR10('../cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class LearnRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = ReLU()  # 小于0设置为0
        self.sigmoid1 = Sigmoid()

    def forward(self, input_data):
        output = self.sigmoid1(input_data)
        return output


learn_relu = LearnRelu()

writer = SummaryWriter('../logs_relu')
for index, data in enumerate(dataloader):
    imgs, targets = data
    output = learn_relu(imgs)
    writer.add_images('input', imgs, index)
    # output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, index)
writer.close()
