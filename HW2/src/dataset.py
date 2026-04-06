import numpy as np
import cv2
import imageio
import torch
from torch.utils.data import Dataset
from skimage.morphology import skeletonize, binary_dilation, diamond


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, size=(512, 512), return_skel=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.return_skel = return_skel

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        if self.return_skel:
            # Tubed skeletonization (CPU, done once per sample during dataloading):
            #   1. binarize  2. extract centerline  3. dilate with diamond(r=2)
            binary = mask.numpy()[0] > 0.5              # (H, W) bool
            skel = skeletonize(binary)
            skel = binary_dilation(skel, footprint=diamond(2))
            skel = torch.from_numpy(skel.astype(np.float32)).unsqueeze(0)  # (1, H, W)
            return image, mask, skel

        return image, mask
