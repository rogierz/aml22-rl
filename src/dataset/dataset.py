"""
This file contains a torch.utils.data.Dataset subclass to use for pytorch dataloaders.
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class HopperDataset(Dataset):
    def __init__(self, annotations_file, split="train", transform=None, target_transform=None, sliding_window=5):
        """
        HopperDataset is a subclass of torch.utils.data.Dataset.

        :param annotations_file: Specify the path to the csv file which contains the mapping <filename>:<observation>
        :param split: Specify which split of the data to use
        :param transform: Specify the data augmentation
        :param target_transform: Specify the transformation on the target data
        :param sliding_window: Define the size of the sliding window used to infer the velocities
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of [train, val, test]")

        csv_data = pd.read_csv(annotations_file)
        self.img_labels = csv_data.loc[csv_data["split"] == split]
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
        label = self.img_labels.iloc[idx+self.sliding_window, 2:]

        if self.target_transform:
            label = self.target_transform(label)

        label = label.to_numpy().astype(float)
        return torch.stack(window), label
