import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from PIL import Image


class MyData(Dataset):

    def __init__(self, root_dir, label_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index) -> T_co:
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        return img, self.label_dir

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = "dataset/train"
    ants_dataset = MyData(root_dir, 'ants')
    img, label = ants_dataset[0]
    img.show()
