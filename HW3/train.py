import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

from model import Autoencoder
from dataloader import get_dataloader, to_img
from concat_image import tensor_to_numpy, add_title, save_comparison_image

def train_autoencoder(config):
    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directories
    save_dirs = {
        'checkpoints': Path(f'./checkpoints/{timestamp}'),
        'images': Path(f'./dc_img/{timestamp}'),
        'logs': Path(f'./logs/{timestamp}')
    }
    
    for dir_path in save_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dirs['logs'] / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loader
    train_loader = get_dataloader(
        config['train_dir'],
        config['batch_size']
    )
    print(f"Total training images: {len(train_loader.dataset)}")
    
    # Initialize model
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # For tracking best model
    best_loss = float('inf')
    best_epoch = 0
    train_losses = []
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (img, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")):
            img = img.to(device)
            
            # Forward pass
            output = model(img)
            loss = criterion(output, img)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Print batch progress
            if batch_idx % 10 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # Print epoch progress
        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Average Loss: {avg_loss:.4f}, '
              f'Time: {elapsed_time/60:.2f} min')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint_path = save_dirs['checkpoints'] / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, checkpoint_path)
            print(f'New best model saved with loss: {best_loss:.4f}')
        
        # Save comparison images periodically
        if epoch % 10 == 0:
            with torch.no_grad():
                # Get the first image from batch
                orig_img = to_img(img[0].cpu())[0]
                recon_img = to_img(output[0].cpu())[0]
                
                # Save comparison image
                save_comparison_image(
                    original=orig_img,
                    reconstructed=recon_img,
                    epoch=epoch,
                    phase='Training',
                    save_dir=str(save_dirs['images'])
                )
        
        # Save loss log
        with open(save_dirs['logs'] / 'loss_log.txt', 'a') as f:
            f.write(f'Epoch {epoch+1}: {avg_loss:.4f}\n')
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, save_dirs['checkpoints'] / 'final_model.pth')
    
    # Training summary
    total_time = time.time() - start_time
    print(f'\nTraining finished!')
    print(f'Best model was saved at epoch {best_epoch} with loss {best_loss:.4f}')
    print(f'Total training time: {total_time/60:.2f} minutes')
    
    # Save training summary
    with open(save_dirs['logs'] / 'training_summary.txt', 'w') as f:
        f.write(f'Training completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Total epochs: {config["num_epochs"]}\n')
        f.write(f'Best epoch: {best_epoch}\n')
        f.write(f'Best loss: {best_loss:.4f}\n')
        f.write(f'Total training time: {total_time/60:.2f} minutes\n')
        f.write(f'Final loss: {avg_loss:.4f}\n')

if __name__ == "__main__":
    config = {
        'train_dir': './data/train/image',
        'num_epochs': 201,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5
    }
    
    train_autoencoder(config)