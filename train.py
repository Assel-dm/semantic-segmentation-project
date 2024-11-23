import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model.unet import build_unet
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
import wandb

#Define the dataset class

# Initialize wandb
wandb.init(project="semantic-segmentation", name="polyp_segmentation", config={
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "optimizer": "Adam",
    "loss_function": "BCEWithLogitsLoss"
})
class PolyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        print(f"Accessing Image: {self.image_paths[idx]}, Mask: {self.mask_paths[idx]}")
        image = cv2.imread(image_path)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Check for valid image and mask
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")
        
        # Normalize mask to [0, 1]
        mask = mask / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask

#Add augmentations using albumentations

def main():
    # Define dataset paths
    image_dir = "data/train"
    mask_dir = "data/train_gt"

    # Ensure the "checkpoints" directory exists
    os.makedirs("checkpoints", exist_ok=True)

    # Generate file paths
    train_image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    train_mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    print("Sample image paths:", train_image_paths[:5])
    print("Sample mask paths:", train_mask_paths[:5])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split dataset into training and validation sets
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        train_image_paths, train_mask_paths, test_size=0.2, random_state=42
    )

    # Define transformations
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    # Create datasets and DataLoaders
    train_dataset = PolyDataset(train_image_paths, train_mask_paths, transform=train_transform)
    val_dataset = PolyDataset(val_image_paths, val_mask_paths, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Define model, optimizer, and loss function
    model = build_unet()
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    # Set number of epochs
    epochs = 30

    # Initialize best validation loss
    best_val_loss = float('inf')

    

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            # Ensure masks have the same shape as output
            optimizer.zero_grad()               # Clear previous gradients
            masks = masks.unsqueeze(1).float()          # Add channel dimension, shape: [B, 1, H, W]
            outputs = model(images)             # Forward pass
            loss = criterion(outputs, masks)    # Compute loss
            loss.backward()                     # Backward pass
            optimizer.step()                    # Update weights
            train_loss += loss.item()           # Accumulate loss for reporting

        train_loss /= len(train_loader)  # Average loss over the epoch
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")


        # Validation loop
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for images, masks in val_loader:
                masks = masks.unsqueeze(1).float()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)  # Average validation loss over the epoch
        print(f"Validation Loss: {val_loss:.4f}")

         # Log losses to wandb
        wandb.log({"epoch": epoch + 1,"train_loss": train_loss,"val_loss": val_loss
    })   


        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()
