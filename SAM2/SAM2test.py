from ultralytics import SAM

model = SAM("sam2.1_b.pt")
trial = model("SAM2/raw_data/bus.jpg")
for tr in trial:
     tr.save("annotated_images/annotated_image.jpg")

ROOT_DIR = 'Cut-and-Paste/raw_data'
import glob 

img_list = [f for f in glob.glob(ROOT_DIR + '/*/*.jpg')]


from pathlib import Path
import os

os.makedirs("annotated_images", exist_ok=True)

for img_path in img_list:
    # 1) Run inference
    results = model(img_path)[0]      # YOLO returns a list with one Results per image

    # 2) Get an annotated image (NumPy array, BGR)
    annotated = results.plot()        # results.plot() draws boxes/masks

    # 3) Build a descriptive filename
    stem = Path(img_path).stem        # e.g. "photo_123"
    save_path = Path("SAM2/annotated_images") / f"{stem}_annotated.jpg"

    # 4) Save with OpenCV (or PIL)
    import cv2
    cv2.imwrite(str(save_path), annotated)

    print(f"Saved annotated image for {img_path} → {save_path}")
