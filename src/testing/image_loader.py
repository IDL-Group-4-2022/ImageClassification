# %%
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import cv2
from PIL import Image

# %%
class CustomImageDataSet(Dataset):
    def __init__(self, df, img_dir, transform=None, target_transform=None, grayscale_transform=None):
        self.df = df
        self.img_dir = img_dir

        self.image_names = np.array(df.index)
        self.transform = transform
        self.target_transform = target_transform
        self.grayscale_transform = grayscale_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, f"im{img_name}.jpg")
        label = np.array(self.df.iloc[[idx]])[0]
        image = cv2.imread(img_path)
        # print(image.shape)
        # print(image.shape[2])


        if image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = Image.fromarray(image)

        if self.transform:
            # print(type(image))
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        # print(type(image))
        # label_tensor = torch.tensor(label, dtype=torch.int64)
        # image = Image.fromarray(image)
        return image, label