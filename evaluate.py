import torch
import numpy as np
from model.unet import build_unet
from train import PolyDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

def dice_coefficient(pred, target):
    """
    Compute Dice coefficient between predicted and target masks.
    """
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)  # Add epsilon to avoid division by zero
    return dice.item()

def mean_dice_coefficient(model, data_loader, device):
    """
    Evaluate the model on a dataset using the mean Dice coefficient.
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, masks in data_loader:
            if images is None or masks is None:
                continue
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = (outputs > 0.2).float()  # Threshold logits to binary predictions
            
            # Compute Dice for each sample in the batch
            for pred, mask in zip(preds, masks):
                dice = dice_coefficient(pred, mask)
                dice_scores.append(dice)

    return np.mean(dice_scores)

def main():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    # Load dataset
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    val_dataset = PolyDataset(
    sorted([os.path.join("data/train", fname) for fname in os.listdir("data/train")]),
    sorted([os.path.join("data/train_gt", fname) for fname in os.listdir("data/train_gt")]),
    transform=val_transform
)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)  # Use num_workers=0 for debugging
    print("Image paths in dataset:", val_dataset.image_paths[:5])
    print("Mask paths in dataset:", val_dataset.mask_paths[:5])
    # Evaluate
    mean_dice = mean_dice_coefficient(model, val_loader, device)
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")

if __name__ == "__main__":
    main()
