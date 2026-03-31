import numpy as np
import cv2
import imageio
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, size=(512, 512)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        image = (image / 255.0).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))          # (3, H, W)
        image = torch.from_numpy(image)

        mask = imageio.v2.imread(self.masks_path[index])
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)             # (1, H, W)
        mask = torch.from_numpy(mask)

        return image, mask
