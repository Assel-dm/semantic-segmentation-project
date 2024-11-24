import numpy as np
import pandas as pd
import cv2
import os

def rle_to_string(runs):
    """Convert a list of RLE runs to a string."""
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    """
    Encode a binary mask into RLE format.

    Args:
        mask (numpy array): Binary mask (values are 0 or 1).
    
    Returns:
        str: RLE-encoded string.
    """
    pixels = mask.flatten()
    pixels[pixels > 0] = 255  # Ensure binary mask values are 0 or 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def process_masks(mask_dir, original_image_dir):
    """
    Process the masks in the specified directory, resize them back to the original
    dimensions, and encode them to RLE.

    Args:
        mask_dir (str): Directory containing the predicted masks.
        original_image_dir (str): Directory containing the original input images.

    Returns:
        dict: Contains 'ids' and 'strings' for the submission.
    """
    strings = []
    ids = []

    for mask_file in os.listdir(mask_dir):  # Loop through each mask file in the directory
        # Extract the image ID from the filename (remove extension)
        image_id = os.path.splitext(mask_file)[0]
        # Remove the prefix "logit_" if present in the mask file name
        original_image_id = image_id.replace("logit_", "")
        
        # Paths for the predicted mask and corresponding original image
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Try both .jpeg and .jpg extensions
        original_image_path_jpeg = os.path.join(original_image_dir, f"{original_image_id}.jpeg")
        original_image_path_jpg = os.path.join(original_image_dir, f"{original_image_id}.jpg")

        # Determine which file exists
        if os.path.exists(original_image_path_jpeg):
            original_image_path = original_image_path_jpeg
        elif os.path.exists(original_image_path_jpg):
            original_image_path = original_image_path_jpg
        else:
            print(f"Error: Original image not found: {original_image_path_jpeg} or {original_image_path_jpg}")
            continue

        print(f"Processing mask: {mask_path}")
        print(f"Using original image: {original_image_path}")

        # Read predicted mask and original image
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Error reading mask file: {mask_path}")
            continue

        original_image = cv2.imread(original_image_path)
        if original_image is None:
            print(f"Error reading original image: {original_image_path}")
            continue

        # Resize mask to original image size
        original_height, original_width = original_image.shape[:2]
        resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Ensure mask is 2D (single-channel)
        if len(resized_mask.shape) == 3:
            resized_mask = resized_mask[:, :, 0]  # Take the first channel

        # Process predictions for both class 0 and class 1
        ids.append(f'{image_id}_0')  # Class 0
        encoded_string = rle_encode_one_mask(resized_mask)
        strings.append(encoded_string)

        ids.append(f'{image_id}_1')  # Class 1
        strings.append("")  # Empty RLE string for class 1

    return {
        'ids': ids,
        'strings': strings,
    }




def create_submission_csv(mask_dir, original_image_dir, sample_submission_path, output_csv_path):
    """
    Create a submission CSV file with RLE-encoded masks resized to the original dimensions,
    ensuring all required `Id` values are included.

    Args:
        mask_dir (str): Directory containing the predicted masks.
        original_image_dir (str): Directory containing the original input images.
        sample_submission_path (str): Path to the sample submission file.
        output_csv_path (str): Path to save the output CSV file.
    """
    # Load expected IDs from the sample submission
    sample_submission = pd.read_csv(sample_submission_path)
    expected_ids = sample_submission["Id"].tolist()

    # Process masks and generate predictions
    results = process_masks(mask_dir, original_image_dir)

    # Map predictions to the expected IDs
    id_to_rle = dict(zip(results["ids"], results["strings"]))

    # Ensure all required IDs are present in the output
    final_submission = []
    for expected_id in expected_ids:
        rle = id_to_rle.get(expected_id, "")  # Use empty string if ID is missing
        final_submission.append({"Id": expected_id, "Expected": rle})

    # Save the final submission file
    final_submission_df = pd.DataFrame(final_submission)
    final_submission_df.to_csv(output_csv_path, index=False)
    print(f"Submission file saved at: {output_csv_path}")


# Specify the paths
MASK_DIR_PATH = "results"  # Directory with predicted masks
ORIGINAL_IMAGE_DIR = "data/test/"  # Directory with original input images
OUTPUT_CSV_PATH = "submission.csv"  # Output CSV file path
SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"
# Create the submission file
create_submission_csv(MASK_DIR_PATH, ORIGINAL_IMAGE_DIR,SAMPLE_SUBMISSION_PATH, OUTPUT_CSV_PATH)
