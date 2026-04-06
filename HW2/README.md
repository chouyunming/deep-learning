# HW2 Project Summary: Retinal Vessel Segmentation on DRIVE Dataset

## 1. Project Overview

This project implements and compares multiple U-Net-based architectures for **retinal blood vessel segmentation** using the [DRIVE dataset](https://drive.grand-challenge.org/). The goal is binary pixel-wise segmentation of vessel structures from fundus (retinal) images. The project also explores advanced loss functions, including a Skeleton-Recall loss designed to improve thin-vessel connectivity.

## 2. Dataset

| Property | Detail |
|---|---|
| **Dataset** | DRIVE (Digital Retinal Images for Vessel Extraction) |
| **Training images** | 20 (indices 21-40) |
| **Test images** | 20 (indices 01-20) |
| **Original format** | `.tif` images + `.gif` manual annotations |
| **Preprocessed format** | `.png` (resized to 512x512) stored in `new_data/` |
| **Annotations** | 1st manual segmentation (binary vessel masks) |

A preprocessing script (`new_data/data_process.py`) converts the raw DRIVE `.tif`/`.gif` files into resized 512x512 `.png` images and masks under `new_data/train/` and `new_data/test/`.

## 3. Network Architectures

Four architectures are implemented in `src/network.py`, all inheriting from a base **UNet** class:

### 3.1 UNet (Baseline)

Standard encoder-decoder with skip connections. Encoder: 4 stages of double 3x3 convolutions (64 -> 128 -> 256 -> 512 channels) with max-pooling. Decoder: bilinear upsampling + concatenation with skip features + double convolutions. Output: 1-channel logit map.

### 3.2 TransUNet

Extends UNet by inserting a **Transformer encoder at the bottleneck**. After the CNN encoder produces a 512-channel feature map at 1/8 resolution, it is flattened into a token sequence, augmented with learned positional embeddings, and processed by a stack of Transformer encoder layers (default: 4 layers, 8 heads). The output is reshaped back to spatial form before the decoder.

### 3.3 Attention U-Net (AttnUNet)

Adds **attention gates** on every skip connection. Before concatenation, encoder features are recalibrated by a gating signal from the decoder path, suppressing irrelevant spatial activations. Uses intermediate channel dimensions of F_int = F_l / 2.

### 3.4 Recurrent Residual U-Net (R2UNet)

Replaces every double-conv block with an **RRCNN block**: a 1x1 projection followed by two stacked Recurrent Blocks (each iterating `t=2` times) with a residual connection. The forward pass is fully inherited from UNet since the module interfaces are preserved.

### Supporting Modules

| Module | Description |
|---|---|
| `double_conv` | Two 3x3 Conv + ReLU layers |
| `Recurrent_block` | Conv-BN-ReLU iterated `t` times with additive feedback |
| `RRCNN_block` | 1x1 Conv + 2x Recurrent_block + residual |
| `Attention_block` | Dual-path gating: W_g(g) + W_x(x) -> sigmoid attention map |

## 4. Loss Functions

Implemented in `src/losses.py`:

| Loss | Formula | Description |
|---|---|---|
| **DiceLoss** | 1 - 2\|P∩G\| / (\|P\| + \|G\|) | Overlap-based; handles class imbalance |
| **DiceBCELoss** | DiceLoss + BCE | Combines region-based and pixel-wise losses |
| **SoftSkeletonRecallLoss** | 1 - (P * Skel).sum / Skel.sum | Penalises missed skeleton pixels; improves thin-vessel recall |
| **DC_SkelREC_and_CE_loss** | w_ce * BCE + w_dice * Dice + w_srec * SkelRecall | Compound loss with configurable weights (default all 1.0) |

The Skeleton-Recall loss requires a precomputed skeleton mask per sample. Skeletonization is performed on-the-fly in the `DriveDataset` class using `skimage.morphology.skeletonize` followed by diamond dilation (radius=2) to create a "tubed" skeleton.

## 5. Training Configuration

| Parameter | Value |
|---|---|
| **Image size** | 512 x 512 |
| **Batch size** | 4 |
| **Epochs** | 1000 |
| **Learning rate** | 1e-4 |
| **Optimizer** | Adam |
| **Validation split** | 4 images (held out from 20 training images) |
| **Random seed** | 42 |
| **Device** | CUDA (if available) |

Training saves:
- `checkpoint.pth` — latest model state
- `best_model.pth` — model with lowest validation loss
- `loss_curve.png` — training/validation loss plot
- `prediction_epoch_N.png` — visual predictions every 100 epochs
- `train_losses.npy` / `val_losses.npy` — raw loss arrays

## 6. Evaluation Metrics & Results (UNet Baseline)

Testing is performed in `src/test.py` using the best checkpoint. Each test image is evaluated independently and results are saved to `results/individual_metrics.csv`.


## 7. Project Structure

```
HW2/
├── DRIVE/                      # Raw DRIVE dataset
│   ├── training/
│   │   ├── images/             # 20 training .tif images (21-40)
│   │   ├── 1st_manual/         # 20 manual annotation .gif masks
│   │   └── mask/               # 20 FOV masks
│   └── test/
│       ├── images/             # 20 test .tif images (01-20)
│       └── mask/               # 20 FOV masks
├── new_data/                   # Preprocessed 512x512 PNG data
│   ├── data_process.py         # DRIVE -> PNG preprocessing script
│   ├── train/
│   │   ├── image/              # 20 training images
│   │   └── 1st_manual/         # 20 training masks
│   └── test/
│       ├── image/              # 20 test images
│       └── 1st_manual/         # 20 test masks
├── src/
│   ├── network.py              # UNet, TransUNet, AttnUNet, R2UNet
│   ├── dataset.py              # DriveDataset (with optional skeletonization)
│   ├── losses.py               # Dice, DiceBCE, SkeletonRecall, compound loss
│   ├── train.py                # Training loop with checkpointing & visualization
│   ├── test.py                 # Evaluation with per-image metrics
│   └── utils.py                # Seeding, directory creation, timing
├── files/
│   └── UNet/                   # Training outputs
│       ├── best_model.pth      # Best model checkpoint
│       ├── loss_curve.png       # Training/validation loss plot
│       └── prediction_epoch_1000.png  # Visual predictions at epoch 1000
├── results/
│   └── UNet/
│       ├── Test/               # Side-by-side test prediction images
│       └── individual_metrics.csv  # Per-image evaluation metrics
├── environment.yaml            # Conda environment specification
├── requirements.txt            # pip dependencies
├── AttnU-Net.pdf               # Reference paper
├── FSG-Net.pdf                 # Reference paper
├── R2U-Net.pdf                 # Reference paper
├── TransUNet.pdf               # Reference paper
├── UNet++.pdf                  # Reference paper
└── skeleton-loss.pdf           # Reference paper (Skeleton-Recall loss)
```

## 8. Dependencies

- Python 3.10
- PyTorch (+ torchvision)
- NumPy, OpenCV, Matplotlib, imageio
- scikit-learn (metrics), scikit-image (skeletonization)
- pandas, tqdm

## 9. Reference Papers

| Paper | Topic |
|---|---|
| **TransUNet** | Transformer-augmented U-Net for medical image segmentation |
| **Attention U-Net** | Attention gates for skip connections |
| **R2U-Net** | Recurrent Residual Convolutional Neural Network |
| **UNet++** | Nested and dense skip connections |
| **FSG-Net** | Fine-grained Semantics-aware Graph matching |
| **Skeleton-Recall** | Skeleton-Recall loss for improved topology in segmentation |

## 10. Usage

```bash
# Preprocess DRIVE data
cd new_data && python data_process.py

# Train with default DiceBCE loss
cd src && python train.py

# Train with Skeleton-Recall compound loss
cd src && python train.py --loss skel_rec --weight_ce 1.0 --weight_dice 1.0 --weight_srec 1.0

# Evaluate on test set
cd src && python test.py
```
