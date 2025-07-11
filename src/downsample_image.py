import cv2
import glob
import os

ROOT_DIR = 'data/raw/object_raw'

TARGET_MAX_DIM = 512

img_list = [
        f for f in glob.glob(ROOT_DIR + '/*/*.jpg')
        if not f.endswith('_mask.jpg') and not f.endswith('_ds.jpg')
    ]

for img_path in img_list:
    # Load image
    image = cv2.imread(img_path)
    print(f"Processing {img_path} with shape {image.shape}")

    # Downsample by scale factor
    h, w = image.shape[:2]
    if max(h, w) > TARGET_MAX_DIM:
        scale = TARGET_MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(img_path.replace('.jpg', '_ds.jpg'), downsampled)
        os.remove(img_path)

        print("Downsampled.")
    else:
        print("Skipped.")
