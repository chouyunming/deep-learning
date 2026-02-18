import torch
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
import os

from model import Autoencoder
from dataloader import get_dataloader, to_img
from concat_image import tensor_to_numpy, add_title, save_comparison_image

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 1.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def test_autoencoder(config):
    # Create output directories
    out_dir = Path('./test_results')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Autoencoder().to(device)
    
    # Load best weights
    checkpoint_path = Path('./checkpoints/20241126_225242/best_model.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load test dataset
    test_loader = get_dataloader(
        config['test_dir'],
        batch_size=1,
        is_test=True
    )
    
    # Testing loop
    total_psnr = 0
    results = []
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for idx, (img, path) in enumerate(tqdm(test_loader, desc="Processing images")):
            img = img.to(device)
            
            # Generate reconstruction
            output = model(img)
            
            # Convert to image format
            original_img = to_img(img)
            reconstructed_img = to_img(output)
            
            # Calculate PSNR
            psnr_value = calculate_psnr(original_img[0], reconstructed_img[0])
            total_psnr += psnr_value
            
            # Save results
            image_name = Path(path[0]).stem
            results.append({
                'path': path[0],
                'psnr': psnr_value
            })
            
            # Save comparison image using your concat_image package
            save_comparison_image(
                original=original_img[0],
                reconstructed=reconstructed_img[0],
                epoch=image_name,
                phase='Test',
                save_dir='./test_results'
            )
    
    # Calculate average PSNR
    avg_psnr = total_psnr / len(test_loader)
    print(f"\nTest Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Sort results by PSNR
    results.sort(key=lambda x: x['psnr'])
    
    # Display worst and best cases
    print("\nWorst 5 reconstructions:")
    for result in results[:5]:
        print(f"File: {Path(result['path']).name}, PSNR: {result['psnr']:.2f} dB")
    
    print("\nBest 5 reconstructions:")
    for result in results[-5:]:
        print(f"File: {Path(result['path']).name}, PSNR: {result['psnr']:.2f} dB")
    
    # Save detailed results
    with open(out_dir / 'test_results.txt', 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n\n")
        f.write("Detailed Results:\n")
        for result in results:
            f.write(f"File: {Path(result['path']).name}, PSNR: {result['psnr']:.2f} dB\n")

if __name__ == "__main__":
    config = {
        'test_dir': './data/test/image',
        'checkpoint_path': './checkpoints/20241126_225242/best_model.pth'
    }
    
    test_autoencoder(config)