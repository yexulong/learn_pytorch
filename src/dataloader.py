import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
test_set = torchvision.datasets.CIFAR10(root='../cifar10_dataset', train=False, transform=dataset_transform,
                                        download=True)

test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter('../dataloader')
for index, data in enumerate(test_loader):
    imgs, targets = data
    writer.add_images('test_data', imgs, index)

writer.close()
