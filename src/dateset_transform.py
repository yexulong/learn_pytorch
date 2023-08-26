import torchvision
from torch.utils.tensorboard import SummaryWriter

 
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='../cifar10_dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='../cifar10_dataset', train=False, transform=dataset_transform, download=True)

# test_set[0][0].show()
writer = SummaryWriter('../P10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()
