from ultralytics import SAM
import numpy as np
import cv2
import glob
from PIL import Image
import os


##################### image segmentation #####################


# Load a model
model = SAM("sam2.1_b.pt")


ROOT_DIR = 'datasets_for_cnp/objects_without_masks'

img_list = [
        f for f in glob.glob(ROOT_DIR + '/*/*.jpg')
        if not f.endswith('_mask.jpg')
    ]

# Run inference with bboxes prompt
for img_path in img_list:
    with Image.open(img_path) as img:
        width, height = img.size

        results = model(
            img_path, 
            bboxes=[[0, 0, width, height]]  # Example bounding box covering the whole image
            )

        # Combine all masks into one binary mask
        masks = results[0].masks.data.cpu().numpy()  # shape: (N, H, W)
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

        cv2.imwrite(
            img_path.replace('.jpg', '_mask.jpg'),
            combined_mask)


##################### invert masks #####################


def should_invert_mask(mask):
    h, w = mask.shape
    corners = [
        mask[0, 0],          # top-left
        mask[0, w-1],        # top-right
        mask[h-1, 0],        # bottom-left
        mask[h-1, w-1],      # bottom-right
    ]
    # Count how many corners are "black" (you can use a threshold if needed)
    black_corners = sum(c <= 10 for c in corners)  # threshold of 10 to allow for slight artifacts
    return black_corners >= 2

def invert_mask_if_needed(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read {mask_path}")

    print("Processing mask at:", mask_path)
    if should_invert_mask(mask):
        mask = 255 - mask  # invert
        print("Inverted mask.")
        #save_path = mask_path.replace('_mask.jpg', '_inverted_mask.jpg')
        cv2.imwrite(mask_path, mask)
    else:
        print("No inversion needed.")
    


mask_list = [
        f for f in glob.glob(ROOT_DIR + '/my_mug/*.jpg')
        if f.endswith('_mask.jpg')
    ]

# Run inference with bboxes prompt
for mask_path in mask_list:
    invert_mask_if_needed(mask_path)
