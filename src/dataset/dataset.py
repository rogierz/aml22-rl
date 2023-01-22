import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd


class HopperDataset(Dataset):
    def __init__(self, annotations_file, split="train", transform=None, target_transform=None, sliding_window=5):
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of [train, val, test]")

        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = os.path.join(os.path.dirname(annotations_file), split)
        self.transform = transform
        self.target_transform = target_transform
        self.sliding_window = sliding_window

    def __len__(self):
        return len(self.img_labels) - self.sliding_window

    def __getitem__(self, idx):
        window = []
        for _ in range(self.sliding_window):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            window.append(image)
        label = self.img_labels.iloc[idx+self.sliding_window, 1:]

        if self.target_transform:
            label = self.target_transform(label)

        label = label.to_numpy().astype(float)
        return torch.stack(window), label
