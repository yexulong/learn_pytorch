# P31 利用GPU训练(二)
# 准备数据集
import time

import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0')
train_data = torchvision.datasets.CIFAR10('../cifar10_dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10('../cifar10_dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)

print(f'训练数据集的长度为: {train_data_size}')
print(f'测试数据集的长度为: {test_data_size}')

# 用DataLoader载入数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class LearnSeq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


learn_seq = LearnSeq()
learn_seq = learn_seq.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(learn_seq.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('../logs_train')
start_time = time.time()
for i in range(epoch):
    print(f'------第 {i + 1} 轮训练开始------')
    # 训练开始
    learn_seq.train()
    for index, datas in enumerate(train_dataloader):
        imgs, targets = datas
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = learn_seq(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f'训练次数: {total_train_step}, Loss: {loss.item()}')
            print(f'耗时: {end_time - start_time}s')
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    learn_seq.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = learn_seq(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f'整体测试集上的Loss: {total_test_loss}')
    print(f'整体测试集上的正确率: {total_accuracy / test_data_size}')
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(learn_seq, f'train/train_{i}.pth')
writer.close()
