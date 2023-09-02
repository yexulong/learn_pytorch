# P32 模型验证
import torch
import torchvision
from PIL import Image
from nn_seq import LearnSeq


device = torch.device('cuda')
image_path = '../imgs/test2.jpg'

image = Image.open(image_path).convert('RGB')
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)
# print(image.shape)
model = torch.load('train/train_9.pth')
# print(model)

image = torch.reshape(image, (1, 3, 32, 32)).to(device)
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

print(output.argmax(1))
