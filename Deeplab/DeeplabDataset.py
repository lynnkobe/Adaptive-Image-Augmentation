from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
from PIL import Image


class Dataset(Dataset):
    def __init__(self, img_folder, label_folder):
        self.img_folder = img_folder
        self.label_folder = label_folder
        img_names = os.listdir(self.img_folder)
        self.img_names = sorted(img_names)

        label_names = os.listdir(self.label_folder)
        self.label_names = sorted(label_names)

    def __len__(self):
        return len(os.listdir(self.img_folder))

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # 归一化输入
        img = torch.Tensor(img)  # 转为Tensor可执行permute操作
        img = img.permute(2, 0, 1)  # 将图片的维度转换成网络输入的维度（channel, width, height）

        label_name = self.label_names[idx]
        label_path = os.path.join(self.label_folder, label_name)
        label = cv2.imread(label_path, 0)
        label = cv2.resize(label, (224, 224))
        label = label / 255.0  # 归一化输入
        label = torch.Tensor(label)  # 转为Tensor可执行permute操作

        return img, label


# 注意这里input的是3 224 224 还是 224 224 3
if __name__ == '__main__':
    dataset = Dataset('AppleRustSet/Train(origin)', 'AppleRustSet/Train_label(origin)')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        # 在这里执行训练操作，使用 inputs 和 labels
        # inputs 的 shape 为 (batch_size, channels, height, width)
        # labels 的 shape 为 (batch_size, num_classes)
        print(1)


