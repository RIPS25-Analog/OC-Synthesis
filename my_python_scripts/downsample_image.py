import cv2
import glob
import os

ROOT_DIR = 'datasets_for_cnp/objects_without_masks'

img_list = [
        f for f in glob.glob(ROOT_DIR + '/*/*.jpg')
        if not f.endswith('_mask.jpg') and not f.endswith('_ds.jpg')
    ]

for img_path in img_list:
    # Load image
    image = cv2.imread(img_path)
    print(f"Processing {img_path} with shape {image.shape}")

    # Downsample by scale factor
    factor = 0.25
    downsampled = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    # Save or display
    cv2.imwrite(img_path.replace('.jpg', '_ds.jpg'), 
                downsampled)

    os.remove(img_path)
