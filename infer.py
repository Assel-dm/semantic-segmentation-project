import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import torch
from model.unet import build_unet
import train
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str, help="Path to the folder containing input images")
parser.add_argument("--output_folder", type=str, default="output_masks", help="Folder to save the segmented masks")


args = parser.parse_args()

# Ensure output folder exists
os.makedirs(args.output_folder, exist_ok=True)

#Load the model and checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_unet().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()


transform = A.Compose([
    A.Resize(256, 256),  # Resize to match the model input size
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ToTensorV2(),  # Convert to PyTorch tensors
])

# Loop through all images in the folder
for image_name in os.listdir(args.input_folder):
    image_path = os.path.join(args.input_folder, image_name)
    output_path = os.path.join(args.output_folder, f"mask_{image_name}")

    # Read and preprocess the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping invalid image: {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_image = transform(image=image)["image"].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)

    # Post-process and save the segmented image
    segmented_image = output.squeeze(0).squeeze(0).cpu().numpy()
    normalized_logits = ((segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min()) * 255).astype('uint8') 
    cv2.imwrite(output_path.replace("mask_", "logit_"), segmented_image)
    cv2.imwrite(output_path.replace("mask_", "logit_"), normalized_logits)
    print(f"Saved segmented mask: {output_path}")





