import argparse
import os
import time
from operator import add
from glob import glob
import numpy as np
import pandas as pd
import cv2
import imageio
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from network import UNet, TransUNet, AttnUNet, R2UNet
from utils import seeding, create_dir


def calculate_metrics(y_true, y_pred):
    y_true = (y_true.cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)
    y_pred = (y_pred.cpu().numpy() > 0.5).astype(np.uint8).reshape(-1)

    return [
        jaccard_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        accuracy_score(y_true, y_pred),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test segmentation model on DRIVE dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pth file (e.g. ../files/unet-dice_bce/best_model.pth)')
    parser.add_argument('--train_patch_size', type=int, default=None,
                        help='Patch size used during training (for TransUNet pos_embed). '
                             'Omit if trained on full images.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ViT-B/16 architecture (768-dim) for TransUNet. '
                             'Must match the flag used during training.')
    args = parser.parse_args()

    seeding(42)

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'new_data')
    test_x = sorted(glob(os.path.join(data_root, 'test', 'image', '*.png')))
    test_y = sorted(glob(os.path.join(data_root, 'test', '1st_manual', '*.png')))

    print(f"Test images: {len(test_x)}  |  Test masks: {len(test_y)}")
    assert len(test_x) == len(test_y), "Mismatch between test images and masks"

    H, W = 512, 512
    size = (W, H)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = args.checkpoint

    # Derive model name from checkpoint directory (format: {model}-{loss})
    run_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    model_name = run_name.split('-', 1)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # For TransUNet, init with training patch size so pos_embed shape matches checkpoint;
    # interpolation in forward() handles the full-image inference resolution.
    transunet_img_size = args.train_patch_size if args.train_patch_size else H
    models = {
        'unet':      UNet(n_class=1),
        'transunet': TransUNet(n_class=1, img_size=transunet_img_size,
                               pretrained=args.pretrained),
        'attnunet':  AttnUNet(n_class=1),
        'r2unet':    R2UNet(n_class=1),
    }
    model = models[model_name].to(device)
    print(f"Model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results_dir = os.path.join(src_dir, '..', 'results', run_name)
    create_dir(results_dir)

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    individual_metrics = {}

    for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.splitext(os.path.basename(x_path))[0]

        image = cv2.imread(x_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        x = torch.from_numpy(
            np.transpose((image / 255.0).astype(np.float32), (2, 0, 1))
        ).unsqueeze(0).to(device)

        raw_mask = imageio.v2.imread(y_path)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
        mask = cv2.resize(raw_mask, size, interpolation=cv2.INTER_NEAREST)
        y = torch.from_numpy(
            np.expand_dims((mask / 255.0).astype(np.float32), axis=(0, 1))
        ).to(device)

        with torch.no_grad():
            t0 = time.time()
            pred = torch.sigmoid(model(x))
            elapsed = time.time() - t0
            time_taken.append(elapsed)

            score = calculate_metrics(y, pred)
            metrics_score = list(map(add, metrics_score, score))

        individual_metrics[name] = {
            'IoU':       score[0],
            'F1':        score[1],
            'Recall':    score[2],
            'Precision': score[3],
            'Accuracy':  score[4],
            'Time_s':    elapsed,
        }

        pred_np = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        gt_np   = mask.astype(np.uint8)
        line    = np.ones((H, 10, 3), dtype=np.uint8) * 128

        def to_rgb(gray):
            return np.stack([gray, gray, gray], axis=-1)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        composite = np.concatenate([image_bgr, line, to_rgb(gt_np), line, to_rgb(pred_np)], axis=1)
        cv2.imwrite(os.path.join(results_dir, f'{name}.png'), composite)

    n = len(test_x)
    mIoU, f1, recall, precision, acc = [s / n for s in metrics_score]

    print(f'\nmIoU: {mIoU:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} '
          f'| Precision: {precision:.4f} | Acc: {acc:.4f}')
    print(f'FPS: {1 / np.mean(time_taken):.2f}')

    os.makedirs(os.path.join(src_dir, '..', 'results'), exist_ok=True)
    pd.DataFrame.from_dict(individual_metrics, orient='index').to_csv(
        os.path.join(src_dir, '..', 'results', 'individual_metrics.csv')
    )
