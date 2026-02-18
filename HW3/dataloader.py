import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, is_test=False):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test
        self.image_files = [f for f in self.img_dir.glob('*') 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, str(img_path)
        return image, 0  # Return 0 as dummy label for training

def get_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])

def get_dataloader(img_dir, batch_size, is_test=False):
    dataset = CustomImageDataset(
        img_dir=img_dir,
        transform=get_transforms(),
        is_test=is_test
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_test,
        num_workers=4,
        pin_memory=True
    )

def to_img(x):
    """Convert tensor to image format"""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x = x * std + mean
    
    # Ensure proper shape and range
    x = x.clamp(0, 1)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x