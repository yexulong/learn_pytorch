import torch
from torch import nn


class LearnNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_data):
        outputdata = input_data + 1
        return outputdata


learn_nn = LearnNN()
x = torch.tensor(1.0)
output = learn_nn(x)
print(output)
