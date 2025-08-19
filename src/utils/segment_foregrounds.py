import os
from ultralytics import SAM
import glob
import cv2
import numpy as np
from PIL import Image
import argparse

def read_darknet_bboxes(bbox_path, image_width, image_height):
	"""Read bounding boxes from darknet format file and convert to pixel coordinates"""
	bboxes = []
	
	with open(bbox_path, 'r') as f:
		for line in f:
			parts = line.strip().split()
			assert len(parts) == 5, f"Invalid bbox line: {line.strip()}"
			
			# Darknet format: class_id x_center y_center width height (normalized)
			x_center, y_center, width, height = map(float, parts[1:])
			
			# Convert from normalized coordinates to pixel coordinates
			x_center_px = x_center * image_width
			y_center_px = y_center * image_height
			width_px = width * image_width
			height_px = height * image_height
			
			# Convert to x1, y1, x2, y2 format
			x1 = int(x_center_px - width_px / 2)
			y1 = int(y_center_px - height_px / 2)
			x2 = int(x_center_px + width_px / 2)
			y2 = int(y_center_px + height_px / 2)
			
			# Ensure coordinates are within image bounds
			x1 = max(0, min(x1, image_width - 1))
			y1 = max(0, min(y1, image_height - 1))
			x2 = max(0, min(x2, image_width - 1))
			y2 = max(0, min(y2, image_height - 1))
			
			bboxes.append([x1, y1, x2, y2])

	return bboxes

def segment_images_from_folder_bbox(class_dir, use_labels=False):
    """
    Segments images in the specified folder using the SAM model with bbox information.
    Assumes class_dir contains two folders: 'images_resized' and 'labels'.
    Each image in 'images_resized' should have a corresponding label file in 'labels' with
    bounding box information in the format: x y w h (where x, y are the
    top-left corner coordinates and w, h are the width and height of the bounding box).
    """

    for image_path in sorted(glob.glob(os.path.join(class_dir, 'images', '*'))):
        image_dimensions = cv2.imread(image_path).shape
        img_name, _ = os.path.splitext(os.path.basename(image_path))

        bboxes = []
        if use_labels:
            bbox_path = os.path.join(class_dir, 'labels', img_name + '.txt')
            bboxes = read_darknet_bboxes(bbox_path, image_dimensions[1], image_dimensions[0])
        
        # Predict segmentation using the SAM model with bounding box
        results = model(image_path, bboxes=bboxes)
        
        masks = results[0].masks

        # Assuming single class segmentation for simplicity, adjust as needed
        mask = masks[0].data.squeeze().cpu().numpy()  # For multi-class, iterate over masks
        mask = mask.astype(np.uint8) # Convert mask to uint8 if needed)
        mask = cv2.resize(mask, (image_dimensions[1], image_dimensions[0]))
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_dimensions[1], image_dimensions[0]))
        
        # Negate the mask
        negative_mask = 1-mask

        # Create subdirectories if they do not exist
        masks_path = os.path.join(class_dir, 'masks')
        if not os.path.exists(masks_path): 
            os.mkdir(masks_path)

        cv2.imwrite(os.path.join(masks_path, img_name + '_mask.png'), negative_mask*255)


if __name__ == "__main__":
    # usage: python src/utils/segment_foregrounds.py /home/data/raw/[dataset_name]/foreground_objects
    parser = argparse.ArgumentParser(description="Segment foreground objects in images.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("--use_labels", action="store_true", help="Use labels for segmentation.")
    args = parser.parse_args()

    model = SAM("sam2.1_l.pt")

    for class_dir in glob.glob(os.path.join(args.dataset_path, "*")):
	    segment_images_from_folder_bbox(class_dir, args.use_labels)