import torch
import torch.nn.functional as F


input_data = torch.tensor([[1, 2, 0, 3, 1],
                           [0, 1, 2, 3, 1],
                           [1, 2, 1, 0, 0],
                           [5, 2, 3, 1, 1],
                           [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input_data = torch.reshape(input_data, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input_data.shape)
print(kernel.shape)

output = F.conv2d(input_data, kernel)
print(output)

output2 = F.conv2d(input_data, kernel, stride=2)
print(output2)

output3 = F.conv2d(input_data, kernel, padding=1)
print(output3)

