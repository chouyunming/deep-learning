import argparse
import os
import time
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import DriveDataset
from network import UNet, TransUNet, AttnUNet
from losses import DiceBCELoss, DC_SkelREC_and_CE_loss
from utils import seeding, create_dir, epoch_time


def train(model, loader, optimizer, loss_fn, device, use_skel=False):
    epoch_loss = 0.0
    model.train()
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(x)
        if use_skel:
            loss = loss_fn(pred, y, batch[2].to(device))
        else:
            loss = loss_fn(pred, y)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, loss_fn, device, use_skel=False):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            if use_skel:
                loss = loss_fn(pred, y, batch[2].to(device))
            else:
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
    parser = argparse.ArgumentParser(description='Train segmentation model on DRIVE dataset')
    parser.add_argument('--model', choices=['unet', 'transunet', 'attnunet'], default='unet',
                        help='Model architecture to train')
    parser.add_argument('--loss', choices=['dice_bce', 'skel_rec'], default='dice_bce',
                        help='Loss function: dice_bce (Dice+BCE) or skel_rec (Dice+SkeletonRecall+BCE)')
    parser.add_argument('--weight_ce',   type=float, default=1.0, help='Weight for BCE term (skel_rec only)')
    parser.add_argument('--weight_dice', type=float, default=1.0, help='Weight for Dice term (skel_rec only)')
    parser.add_argument('--weight_srec', type=float, default=1.0, help='Weight for Skeleton-Recall term (skel_rec only)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size for patch-based training (e.g. 48). Omit for full-image training.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load ViT-B/16 ImageNet pretrained weights for TransUNet')
    parser.add_argument('--clahe', action='store_true',
                        help='Enable CLAHE preprocessing on train + val')
    parser.add_argument('--augment', action='store_true',
                        help='Enable training-time augmentations (flips, rotation, elastic)')
    args = parser.parse_args()

    seeding(42)

    if args.patch_size is not None:
        output_path = os.path.join('files', f'{args.model}-{args.loss}-p{args.patch_size}')
    else:
        output_path = os.path.join('files', f'{args.model}-{args.loss}')
    create_dir(output_path)

    data_root = os.path.join(os.path.dirname(__file__), '..', 'new_data')
    train_x = sorted(glob(os.path.join(data_root, 'train', 'image', '*.png')))
    train_y = sorted(glob(os.path.join(data_root, 'train', '1st_manual', '*.png')))

    print(f"Total training samples: {len(train_x)}")
    assert len(train_x) == len(train_y), "Mismatch between images and masks"

    H, W = 512, 512
    batch_size = args.batch_size
    num_epochs = 1000
    lr = 1e-4
    val_split = 4        # hold out 4 images for validation

    checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
    best_model_path = os.path.join(output_path, 'best_model.pth')

    use_skel = args.loss == 'skel_rec'
    patch_size = args.patch_size

    # Split image paths first, then create datasets with appropriate settings
    n_val = val_split
    n_train = len(train_x) - n_val
    indices = list(range(len(train_x)))
    gen = torch.Generator().manual_seed(42)
    val_indices = torch.randperm(len(train_x), generator=gen)[:n_val].tolist()
    train_indices = [i for i in indices if i not in val_indices]

    train_dataset = DriveDataset(
        [train_x[i] for i in train_indices],
        [train_y[i] for i in train_indices],
        size=(H, W), return_skel=use_skel,
        patch_size=patch_size,
        augment=args.augment, clahe=args.clahe,
    )
    # Validation always uses full images with no augmentation
    val_dataset = DriveDataset(
        [train_x[i] for i in val_indices],
        [train_y[i] for i in val_indices],
        size=(H, W), return_skel=use_skel,
        augment=False, clahe=args.clahe,
    )
    print(f"Preprocessing: CLAHE={args.clahe}  |  Augment={args.augment}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For TransUNet, use patch_size for positional embedding if patch training is enabled
    transunet_img_size = patch_size if patch_size else H
    models = {
        'unet':     UNet(n_class=1),
        'transunet': TransUNet(n_class=1, img_size=transunet_img_size, pretrained=args.pretrained),
        'attnunet': AttnUNet(n_class=1),
    }
    model = models[args.model].to(device)
    print(f"Model: {args.model}")
    if patch_size:
        patches_per_image = (H // patch_size) * (W // patch_size)
        print(f"Patch-based training: {patch_size}x{patch_size}, {patches_per_image} patches/image")
    else:
        print(f"Full-image training: {H}x{W}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_skel:
        loss_fn = DC_SkelREC_and_CE_loss(
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            weight_srec=args.weight_srec,
        )
        print(f"Loss: DC_SkelREC_and_CE  "
              f"(w_ce={args.weight_ce}, w_dice={args.weight_dice}, w_srec={args.weight_srec})")
    else:
        loss_fn = DiceBCELoss()
        print("Loss: DiceBCELoss")

    sample_x, sample_y = next(iter(val_loader))[:2]
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device, use_skel=use_skel)
        val_loss = evaluate(model, val_loader, loss_fn, device, use_skel=use_skel)
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

        if (epoch + 1) % 100 == 0:
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
