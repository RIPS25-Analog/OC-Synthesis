import os
import shutil
import random

# --- CONFIG ---
num_samples = 20  # Number of image-mask pairs to sample
images_dir = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/OI_subset/cat/images"
masks_dir = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/OI_subset/cat/masks"
output_dir = "./synthetic_root/cat"

# --- SETUP ---
os.makedirs(output_dir, exist_ok=True)

# --- SAMPLE ---
all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
selected_images = random.sample(all_images, num_samples)

for img_file in selected_images:
    img_path = os.path.join(images_dir, img_file)
    mask_file = img_file.replace(".jpg", "_mask0.pbm")
    mask_path = os.path.join(masks_dir, mask_file)
    
    if not os.path.exists(mask_path):
        print(f"⚠️ Skipping {img_file}: mask {mask_file} not found.")
        continue

    # Copy image
    shutil.copy(img_path, os.path.join(output_dir, img_file))

    # Copy mask
    shutil.copy(mask_path, os.path.join(output_dir, mask_file))

print(f"✅ Copied {len(selected_images)} image-mask pairs to {output_dir}")
