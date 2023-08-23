# P9 transforms的使用
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'dataset/train/ants/0013035.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image('Tensor_img', tensor_img)

writer.close()
# cv_img = cv2.imread(img_path)
# print()
