# P26-1 模型的保存
import torch
import torchvision

from nn_seq import LearnSeq

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, 'vgg16_method1.pth')

# 保存方式2,模型参数(官方推荐)
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# 陷阱
learn_seq = LearnSeq()
torch.save(learn_seq, 'learn_seq.pth')
