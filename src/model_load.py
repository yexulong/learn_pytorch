# P26-2 模型的加载
import torch
import torchvision

# 读取方式1 -> 对应保存方式1，加载模型
model = torch.load('vgg16_method1.pth')
print(model)

# 读取方式2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16)

# 陷阱1
model = torch.load('learn_seq.pth')
print(model)