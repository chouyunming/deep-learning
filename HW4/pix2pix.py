import numpy as np 
import matplotlib.pyplot as plt 
import os
import json
import copy
import pickle
import math
from PIL import Image
from statistics import mean 
from tqdm import tqdm 

import torch 
import torch.nn as nn 
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, random_split

# # Dataroot
image_dir = './Hw4_GAN_western_blot/wb_dataset'
mask_dir = './Hw4_GAN_western_blot/wb_template'

# # Setting
MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 256
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = batch_size

# Define Dataset
class WesternBlotDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, store_sizes=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.store_sizes = store_sizes
        self.image_sizes = {}
        
        # Get all image filenames and ensure they have corresponding masks
        self.image_files = []
        for f in sorted(os.listdir(image_dir)):
            if f.endswith('.png'):
                mask_name = f.replace('bg_', 'BandMask_bg_')
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    self.image_files.append(f)
                    
                    # Store original image sizes if requested
                    if store_sizes:
                        with Image.open(os.path.join(image_dir, f)) as img:
                            self.image_sizes[f] = img.size
        
        # Save image sizes to JSON if store_sizes is True
        if store_sizes:
            with open('image_sizes.json', 'w') as f:
                json.dump(self.image_sizes, f)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('bg_', 'BandMask_bg_')
        
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Store original size before transformation
        original_size = image.size
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_name,
            'original_size': original_size
        }

def create_data_loaders(image_dir, mask_dir, batch_size=4, train_split=0.8):
    # Create separate transforms for images and masks
    image_transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Single channel normalization
    ])
    
    # Create complete dataset without storing sizes initially
    dataset = WesternBlotDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
        store_sizes=False  # Don't store sizes for the full dataset
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create train loader without storing sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create a new test dataset with size storing enabled
    test_dataset_with_sizes = WesternBlotDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
        store_sizes=True  # Store sizes only for test dataset
    )
    
    # Use the same indices as the original test_dataset
    test_indices = test_dataset.indices
    test_dataset_with_sizes = torch.utils.data.Subset(test_dataset_with_sizes, test_indices)
    
    # Create test loader with the new dataset
    test_loader = DataLoader(
        test_dataset_with_sizes,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader
    
# Create DataLoader
train_loader, test_loader = create_data_loaders(
    image_dir=image_dir,
    mask_dir=mask_dir,
    batch_size=batch_size
)


# Define Generator
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3): 
        super(Generator, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu5 = nn.LeakyReLU(0.2)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.lrelu6 = nn.LeakyReLU(0.2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn7 = nn.BatchNorm2d(512)
        self.lrelu7 = nn.LeakyReLU(0.2)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)
        self.bn8 = nn.BatchNorm2d(512)

        # Decoder 
        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn1 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn2 = nn.BatchNorm2d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.deconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn3 = nn.BatchNorm2d(512)
        self.dropout3 = nn.Dropout(0.5)
        
        self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout(0.5)
        
        self.deconv5 = nn.ConvTranspose2d(1024, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn5 = nn.BatchNorm2d(256)
        self.dropout5 = nn.Dropout(0.5)
        
        self.deconv6 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn6 = nn.BatchNorm2d(128)
        self.dropout6 = nn.Dropout(0.5)
        
        self.deconv7 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dbn7 = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout(0.5)
        
        self.deconv8 = nn.ConvTranspose2d(128, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.lrelu1(self.conv1(x))           # print(f"e1 shape: {e1.shape}")
        e2 = self.lrelu2(self.bn2(self.conv2(e1)))  # print(f"e2 shape: {e2.shape}")
        e3 = self.lrelu3(self.bn3(self.conv3(e2)))  # print(f"e3 shape: {e3.shape}")
        e4 = self.lrelu4(self.bn4(self.conv4(e3)))  # print(f"e4 shape: {e4.shape}")
        e5 = self.lrelu5(self.bn5(self.conv5(e4)))  # print(f"e5 shape: {e5.shape}")
        e6 = self.lrelu6(self.bn6(self.conv6(e5)))  # print(f"e6 shape: {e6.shape}")
        e7 = self.lrelu7(self.bn7(self.conv7(e6)))  # print(f"e7 shape: {e7.shape}")
        e8 = self.bn8(self.conv8(e7))               # print(f"e8 shape: {e8.shape}")
        
        # Decoder with skip connections
        d1 = self.dropout1(self.dbn1(self.deconv1(self.relu(e8))))  # print(f"d1 shape: {d1.shape}")
        d1 = torch.cat([d1, e7], 1)
        
        d2 = self.dropout2(self.dbn2(self.deconv2(self.relu(d1))))  # print(f"d2 shape: {d2.shape}")
        d2 = torch.cat([d2, e6], 1)
        
        d3 = self.dropout3(self.dbn3(self.deconv3(self.relu(d2))))
        d3 = torch.cat([d3, e5], 1)
        
        d4 = self.dropout4(self.dbn4(self.deconv4(self.relu(d3))))
        d4 = torch.cat([d4, e4], 1)
        
        d5 = self.dropout5(self.dbn5(self.deconv5(self.relu(d4))))
        d5 = torch.cat([d5, e3], 1)
        
        d6 = self.dropout6(self.dbn6(self.deconv6(self.relu(d5))))
        d6 = torch.cat([d6, e2], 1)
        
        d7 = self.dropout7(self.dbn7(self.deconv7(self.relu(d6))))
        d7 = torch.cat([d7, e1], 1)
        
        d8 = self.deconv8(self.relu(d7))
        return self.tanh(d8)
    
# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        # Conv2: 128x128 -> 64x64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        # Conv3: 64x64 -> 32x32
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)
        
        # Conv4: 32x32 -> 16x16
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        # Linear層輸出1個值
        self.linear = nn.Linear(16 * 16 * 512, 1)

    def forward(self, img, condition):
        x = torch.cat([img, condition], dim=1)
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)
        return x

# Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples, condition):
    """計算梯度懲罰"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, condition)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Define Training
def train_fn(train_dl, G, D, criterion_bce, criterion_mae, optimizer_g, optimizer_d, epoch):
    G.train()
    D.train()
    LAMBDA = 10.0  # 降低 LAMBDA 值
    GP_LAMBDA = 10.0  # 梯度懲罰的權重
    D_WEIGHT = 0.5  # 新增：鑑別器損失權重
    total_loss_g, total_loss_d = [], []
    running_g_bce = 0.0
    running_g_mae = 0.0
    running_d_real = 0.0
    running_d_fake = 0.0

    for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch}')):
        input_img = batch['mask'].to(device)
        real_img = batch['image'].to(device)
        batch_size = input_img.size(0)

        # 標籤平滑化：調整標籤範圍
        real_label = torch.ones(batch_size, 1, device=device) * 0.9  # 從0.95降到0.9
        fake_label = torch.zeros(batch_size, 1, device=device) * 0.1  # 從0增加到0.1
        
        # ---------- 訓練判別器 ----------
        optimizer_d.zero_grad()
        
        # 降低噪聲強度
        noisy_real = real_img + torch.randn_like(real_img) * 0.1  # 增加噪聲
        out_real = D(noisy_real, input_img)
        loss_d_real = criterion_bce(out_real, real_label)

        fake_img = G(input_img)
        noisy_fake = fake_img.detach() + torch.randn_like(fake_img) * 0.1
        out_fake = D(noisy_fake, input_img)
        loss_d_fake = criterion_bce(out_fake, fake_label)

        gradient_penalty = compute_gradient_penalty(D, noisy_real, noisy_fake, input_img)
        
        # 降低鑑別器損失的權重
        loss_d = (loss_d_real + loss_d_fake + GP_LAMBDA * gradient_penalty) * D_WEIGHT
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)  # 降低梯度裁剪閾值
        optimizer_d.step()

        # ---------- 再訓練生成器 ----------
        optimizer_g.zero_grad()
        
        fake_img = G(input_img)
        noisy_fake = fake_img + torch.randn_like(fake_img) * 0.05
        out_fake = D(noisy_fake, input_img)
        
        loss_g_bce = criterion_bce(out_fake, real_label)
        loss_g_mae = criterion_mae(fake_img, real_img)
        loss_g_mae = loss_g_mae / fake_img.shape[1]
        
        loss_g = loss_g_bce + LAMBDA * loss_g_mae
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optimizer_g.step()

        # 記錄損失
        running_g_bce += loss_g_bce.item()
        running_g_mae += loss_g_mae.item()
        running_d_real += loss_d_real.item()
        running_d_fake += loss_d_fake.item()
        total_loss_g.append(loss_g.item())
        total_loss_d.append(loss_d.item())

        if (i + 1) % 100 == 0:
            print(f"\nBatch {i+1}/{len(train_dl)}:")
            print(f"Generator Loss - BCE: {running_g_bce/100:.4f}, MAE: {running_g_mae/100:.4f}")
            print(f"Discriminator Loss - Real: {running_d_real/100:.4f}, Fake: {running_d_fake/100:.4f}")
            running_g_bce = running_g_mae = running_d_real = running_d_fake = 0.0

    return mean(total_loss_g), mean(total_loss_d), fake_img.detach().cpu()

# Saving Generated Images
def saving_img(fake_img, epoch):
    os.makedirs("generated", exist_ok=True)
    save_image(fake_img, f"generated/fake_epoch_{epoch}.png", value_range=(-1.0, 1.0), normalize=True)

# Saving Training Info 
def saving_logs(result):
    with open("train.pkl", "wb") as f:
        pickle.dump(result, f)

# Saving Checkpoints
def saving_model(D, G, G_avg, epoch):
    os.makedirs("weight", exist_ok=True)
    torch.save({
        'G_state_dict': G.state_dict(),
        'G_avg_state_dict': G_avg.state_dict(),
        'D_state_dict': D.state_dict(),
    }, f"weight/checkpoint_epoch_{epoch}.pth")

# 更新移動平均模型
def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        for param_tgt, param_src in zip(model_tgt.parameters(), model_src.parameters()):
            param_tgt.data = beta * param_tgt.data + (1-beta) * param_src.data

# Plotting Losses
def show_losses(generator_losses, discriminator_losses):
    plt.figure(figsize=(10, 6))
    
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.title("Generator and Discriminator Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    ax = plt.gca() 
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

# Define Training Loop
def train_loop(train_dl, G, D, num_epoch, lr_g=0.0002, lr_d=0.00005, betas=(0.5, 0.999)):
    G.to(device)
    D.to(device)
    
    # 創建移動平均模型
    G_avg = copy.deepcopy(G)
    G_avg.to(device)
    
    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode='min', factor=0.2, patience=3, verbose=True
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode='min', factor=0.2, patience=3, verbose=True
    )
    
    criterion_mae = nn.L1Loss()
    criterion_bce = nn.BCEWithLogitsLoss()
    
    total_loss_d, total_loss_g = [], []
    best_g_loss = float('inf')  # 初始化最佳生成器損失

    print("Start Training...")
    for epoch in range(1, num_epoch + 1):
        loss_g, loss_d, fake_img = train_fn(
            train_dl, G, D, criterion_bce, criterion_mae, 
            optimizer_g, optimizer_d, epoch
        )
        
        scheduler_g.step(loss_g)
        scheduler_d.step(loss_d)
        update_average(G_avg, G)
        
        total_loss_g.append(loss_g)
        total_loss_d.append(loss_d)
        
        print(f"\nEpoch {epoch}/{num_epoch}")
        print(f"Generator Loss: {loss_g:.4f}")
        print(f"Discriminator Loss: {loss_d:.4f}")
        
        saving_img(fake_img, epoch)
        
        # 使用移動平均模型生成圖片
        with torch.no_grad():
            avg_fake_img = G_avg(next(iter(train_dl))['mask'].to(device))
            saving_img(avg_fake_img.cpu(), f"avg_{epoch}")

        if epoch % 5 == 0:
            saving_model(D, G, G_avg, epoch)

        if loss_g < best_g_loss:
            best_g_loss = loss_g
            print(f"Saving best model with loss: {best_g_loss:.4f}")
            saving_model(D, G, G_avg, "best")

    result = {
        "G_loss": total_loss_g,
        "D_loss": total_loss_d
    }
    saving_logs(result)
    show_losses(total_loss_g, total_loss_d)
    saving_model(D, G, G_avg, "final")
    
    print("Training Done!")
    return G, G_avg, D

# Define PSNR
def calculate_psnr(img1, img2):
    # 確保圖像在 [0, 1] 範圍內
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Define Load (best) Model
def load_model():
    G = Generator()
    checkpoint = torch.load("weight/checkpoint_epoch_best.pth", 
                          map_location={"cuda": "cpu"})
    G.load_state_dict(checkpoint['G_state_dict'])
    G.eval()
    return G.to(device)

def train_show_img(name, G):
    root = "generated"
    fig, axes = plt.subplots(int(name), 1, figsize=(12, 18))
    ax = axes.ravel()
    for i in range(int(name)):
        filename = os.path.join(root, f"fake{str(i+1)}.png")
        ax[i].imshow(Image.open(filename))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

def de_norm(img):
    device = img.device
    std_tensor = torch.FloatTensor(STD).view(3, 1, 1).to(device)
    mean_tensor = torch.FloatTensor(MEAN).view(3, 1, 1).to(device)
    
    img_ = img.mul(std_tensor)
    img_ = img_.add(mean_tensor).detach()
    img_ = img_.clamp(min=-1, max=1).permute(1, 2, 0)
    
    return img_.cpu().numpy()

def get_original_dataset(dataset):
    if hasattr(dataset, 'dataset'):
        return get_original_dataset(dataset.dataset)
    return dataset

# Define Evaluate
def evaluate(test_loader, G, num_samples=len(test_loader.dataset), save_dir='test_results', display_samples=None):
    """
    Evaluate generator and display/save results with original size restoration
    """
    os.makedirs(save_dir, exist_ok=True)
    original_dataset = get_original_dataset(test_loader.dataset)
    
    # Load original image sizes
    with open('image_sizes.json', 'r') as f:
        image_sizes = json.load(f)
    
    G.eval()
    total_psnr = 0
    
    with torch.no_grad():
        available_samples = min(num_samples, len(test_loader.dataset))
        display_count = available_samples if display_samples is None else min(display_samples, available_samples)
        
        print(f"Processing {available_samples} samples, displaying {display_count} samples")
        
        if display_count > 0:
            fig, axes = plt.subplots(display_count, 3, figsize=(15, display_count * 5))
            if display_count == 1:
                axes = np.array([axes])
            axes = axes.ravel()
        
        sample_count = 0
        progress_bar = tqdm(total=available_samples, desc="Processing samples")
        
        for i, batch in enumerate(test_loader):
            for j in range(batch['mask'].size(0)):
                if sample_count >= available_samples:
                    break
                    
                input_img = batch['mask'][j:j+1].to(device)
                real_img = batch['image'][j:j+1].to(device)
                filename = batch['filename'][j]
                original_size = image_sizes[filename]
                
                # Generate fake image
                fake_img = G(input_img)
                
                # Convert to numpy and denormalize
                real_np = de_norm(real_img[0])
                fake_np = de_norm(fake_img[0])
                input_np = de_norm(input_img[0])
                
                # Resize back to original dimensions
                real_pil = Image.fromarray((real_np * 255).astype(np.uint8))
                fake_pil = Image.fromarray((fake_np * 255).astype(np.uint8))
                input_pil = Image.fromarray((input_np * 255).astype(np.uint8))
                
                real_resized = np.array(real_pil.resize(original_size)) / 255.0
                fake_resized = np.array(fake_pil.resize(original_size)) / 255.0
                input_resized = np.array(input_pil.resize(original_size)) / 255.0
                
                # Calculate metrics on resized images
                psnr = calculate_psnr(real_resized, fake_resized)
                total_psnr += psnr
                
                # Display samples if needed
                if sample_count < display_count:
                    # Display original size images
                    axes[sample_count * 3].imshow(input_resized, cmap='gray')
                    axes[sample_count * 3].set_title(f"Input Mask", fontsize=12)
                    axes[sample_count * 3].axis('off')

                    axes[sample_count * 3 + 1].imshow(real_resized)
                    axes[sample_count * 3 + 1].set_title("Real Image", fontsize=12)
                    axes[sample_count * 3 + 1].axis('off')

                    axes[sample_count * 3 + 2].imshow(fake_resized)
                    axes[sample_count * 3 + 2].set_title(f"Generated Image\nPSNR: {psnr:.2f}", fontsize=12)
                    axes[sample_count * 3 + 2].axis('off')

                # Save individual result images
                plt.figure(figsize=(15, 5))
                
                plt.subplot(131)
                plt.imshow(input_resized, cmap='gray')
                plt.title(f"Input Mask\n")
                plt.axis('off')
                
                plt.subplot(132)
                plt.imshow(real_resized)
                plt.title("Real Image")
                plt.axis('off')
                
                plt.subplot(133)
                plt.imshow(fake_resized)
                plt.title(f"Generated Image\nPSNR: {psnr:.2f}")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_results.png"))
                plt.close()
                
                # Save metrics
                with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
                    f.write(f'{filename}: PSNR={psnr:.2f}\n')
                
                sample_count += 1
                progress_bar.update(1)
                
            if sample_count >= available_samples:
                break
        
        progress_bar.close()

        # Calculate and display average metrics
        avg_psnr = total_psnr / sample_count
        print(f"\nAverage PSNR: {avg_psnr:.2f}")
        
        # Save average metrics
        with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
            f.write(f'\nAverage Metrics:\nPSNR={avg_psnr:.2f}\n')

if __name__=='__main__':
    G = Generator(in_channels=1, out_channels=3)
    D = Discriminator(in_channels=4)
    num_epochs=200

    # training
    trained_G, _, trained_D = train_loop(train_loader, G, D, num_epochs)

    # testing
    trained_G = load_model()
    evaluate(test_loader, trained_G, save_dir='test_results')