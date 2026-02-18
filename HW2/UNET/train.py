import os
import time
from glob import glob
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data_ae import DriveDataset_AE
from model import build_unet
from utils import seeding, create_dir, epoch_time

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, _ in loader:
        x = x.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = loss_fn(reconstructed, x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/len(loader)

def save_reconstruction(model, x, path, epoch, device):
    model.eval()
    with torch.no_grad():
        x = x[:4].to(device)
        reconstructed = model(x)
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        for i in range(4):
            orig = x[i].cpu().numpy().transpose(1,2,0)
            axes[0,i].imshow(orig)
            axes[0,i].axis('off')
            axes[0,i].set_title('Original')
            
            recon = reconstructed[i].cpu().numpy().transpose(1,2,0)
            axes[1,i].imshow(recon)
            axes[1,i].axis('off')
            axes[1,i].set_title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'reconstruction_epoch_{epoch+1}.png'))
        plt.close()

if __name__ == "__main__":
    seeding(42)

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    path = os.path.join('files', now)
    create_dir(path)

    train_x = sorted(glob("./new_data/train/image/*"))
    print(f"Dataset Size: {len(train_x)}")

    H, W = 512, 512
    batch_size = 8
    num_epochs = 200
    lr = 1e-4
    checkpoint_path = os.path.join(path, "checkpoint.pth")
    best_model_path = os.path.join(path, "best_model.pth")

    train_dataset = DriveDataset_AE(train_x, train_x)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    sample_batch = next(iter(train_loader))[0]
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)

        # 儲存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'best_loss': best_loss,
                'train_losses': train_losses
            }, best_model_path)

        # 儲存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'best_loss': best_loss,
            'train_losses': train_losses
        }, checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n\tTrain Loss: {train_loss:.6f}')
        if train_loss == best_loss:
            print(f'\tNew Best Loss!')

        if (epoch + 1) % 50 == 0:
            save_reconstruction(model, sample_batch, path, epoch, device)

    np.save(os.path.join(path, 'train_losses.npy'), np.array(train_losses))
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss_curve.png'))
    plt.close()