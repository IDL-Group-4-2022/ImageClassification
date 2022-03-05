"""
Generates a custom image dataset for a DataLoader.
"""

# %%
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.labels = df.values
        self.image_names = np.array(df.index)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = Path(self.img_dir) / f"im{img_name}.jpg"

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
