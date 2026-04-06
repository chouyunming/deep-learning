import os
import time
from glob import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from dataset import DriveDataset
from network import UNet
from losses import DiceLoss
from utils import seeding, create_dir, epoch_time


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def save_predictions(model, sample_x, sample_y, path, epoch, device):
    model.eval()
    with torch.no_grad():
        n = min(4, sample_x.size(0))
        x = sample_x[:n].to(device)
        pred = torch.sigmoid(model(x))

        fig, axes = plt.subplots(3, n, figsize=(n * 4, 12))
        for i in range(n):
            axes[0, i].imshow(x[i].cpu().numpy().transpose(1, 2, 0))
            axes[0, i].axis('off')
            axes[0, i].set_title('Input')

            axes[1, i].imshow(sample_y[i].cpu().numpy().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title('Ground Truth')

            axes[2, i].imshow(pred[i].cpu().numpy().squeeze() > 0.5, cmap='gray')
            axes[2, i].axis('off')
            axes[2, i].set_title('Prediction')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'prediction_epoch_{epoch + 1}.png'))
        plt.close()


if __name__ == "__main__":
    seeding(42)

    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = os.path.join('files', now)
    create_dir(output_path)

    data_root = '../new_data'
    train_x = sorted(glob(os.path.join(data_root, 'train', 'image', '*.png')))
    train_y = sorted(glob(os.path.join(data_root, 'train', '1st_manual', '*.png')))

    print(f"Total training samples: {len(train_x)}")
    assert len(train_x) == len(train_y), "Mismatch between images and masks"

    H, W       = 512, 512
    batch_size = 4
    num_epochs = 150
    lr         = 1e-4
    val_split  = 4        # hold out 4 images for validation

    checkpoint_path  = os.path.join(output_path, 'checkpoint.pth')
    best_model_path  = os.path.join(output_path, 'best_model.pth')

    full_dataset = DriveDataset(train_x, train_y, size=(H, W))
    val_dataset, train_dataset = random_split(
        full_dataset, [val_split, len(full_dataset) - val_split],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = UNet(n_class=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = DiceLoss()

    sample_x, sample_y = next(iter(val_loader))
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss   = evaluate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_model_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)

        mins, secs = epoch_time(start_time, time.time())
        marker = ' *' if val_loss == best_val_loss else ''
        print(f'Epoch {epoch + 1:03}/{num_epochs} | {mins}m {secs}s'
              f' | Train: {train_loss:.4f} | Val: {val_loss:.4f}{marker}')

        if (epoch + 1) % 50 == 0:
            save_predictions(model, sample_x, sample_y, output_path, epoch, device)

    np.save(os.path.join(output_path, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(output_path, 'val_losses.npy'),   np.array(val_losses))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.title('DRIVE — Training & Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.close()
