import numpy as np
import cv2
import imageio
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.morphology import skeletonize, binary_dilation, diamond


def apply_clahe(image_uint8, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE on the L channel of LAB; keeps RGB output."""
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def elastic_transform(image, mask, alpha, sigma, rng):
    """Elastic deformation (Simard et al., 2003) applied jointly to image and mask."""
    h, w = image.shape[:2]
    dx = gaussian_filter((rng.random((h, w)) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random((h, w)) * 2 - 1), sigma) * alpha
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = np.stack([(y + dy).ravel(), (x + dx).ravel()], axis=0)

    if image.ndim == 3:
        warped = np.stack(
            [map_coordinates(image[..., c], indices, order=1, mode='reflect').reshape(h, w)
             for c in range(image.shape[2])], axis=-1)
    else:
        warped = map_coordinates(image, indices, order=1, mode='reflect').reshape(h, w)
    warped_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(h, w)
    return warped, warped_mask


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, size=(512, 512), return_skel=False,
                 patch_size=None, augment=False,
                 rotation_range=30, elastic_alpha=40, elastic_sigma=6,
                 elastic_prob=0.5, rotation_prob=0.5, clahe=True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.return_skel = return_skel
        self.patch_size = patch_size
        self.augment = augment
        self.rotation_range = rotation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob
        self.rotation_prob = rotation_prob
        self.clahe = clahe
        if patch_size is not None:
            self.patches_per_image = (size[0] // patch_size) * (size[1] // patch_size)
        else:
            self.patches_per_image = 1

    def __len__(self):
        if self.patch_size is not None:
            return len(self.images_path) * self.patches_per_image
        return len(self.images_path)

    def __getitem__(self, index):
        if self.patch_size is not None:
            img_index = index // self.patches_per_image
        else:
            img_index = index

        image = cv2.imread(self.images_path[img_index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)

        mask = imageio.v2.imread(self.masks_path[img_index])
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # --- Preprocessing (applied to train + val) ---
        if self.clahe:
            image = apply_clahe(image)

        image = (image / 255.0).astype(np.float32)
        mask = (mask / 255.0).astype(np.float32)

        # --- Patch extraction ---
        if self.patch_size is not None:
            h, w = image.shape[:2]
            p = self.patch_size
            top = np.random.randint(0, h - p + 1)
            left = np.random.randint(0, w - p + 1)
            image = image[top:top+p, left:left+p, :]
            mask = mask[top:top+p, left:left+p]

        # --- Train-only augmentations ---
        if self.augment:
            rng = np.random

            if rng.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            if rng.random() > 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

            if rng.random() < self.rotation_prob:
                angle = rng.uniform(-self.rotation_range, self.rotation_range)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT)

            if rng.random() < self.elastic_prob:
                image, mask = elastic_transform(
                    image, mask,
                    alpha=self.elastic_alpha,
                    sigma=self.elastic_sigma,
                    rng=rng,
                )
                image = image.astype(np.float32)
                mask = mask.astype(np.float32)

        image = np.transpose(image, (2, 0, 1))          # (3, H, W)
        image = torch.from_numpy(np.ascontiguousarray(image))

        mask = np.expand_dims(mask, axis=0)             # (1, H, W)
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.return_skel:
            # Tubed skeletonization: binarize → centerline → dilate with diamond(r=2)
            binary = mask.numpy()[0] > 0.5
            skel = skeletonize(binary)
            skel = binary_dilation(skel, footprint=diamond(2))
            skel = torch.from_numpy(skel.astype(np.float32)).unsqueeze(0)
            return image, mask, skel

        return image, mask
