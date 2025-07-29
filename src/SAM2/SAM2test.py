from ultralytics import SAM
from pathlib import Path
import numpy as np, cv2, os, torch, tqdm

# ------------------------------------------------------------------
# 0.  Folder layout (produced by oi_download_dataset --format darknet)
# ------------------------------------------------------------------
ROOT     = Path("OI_subset")        # base_dir you used in the downloader
IMG_DIR  = ROOT / "cat" / "images"
LBL_DIR  = ROOT / "cat" / "darknet"  # *.txt files live here

MASK_DIR = Path("SAM2/masks");            MASK_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR  = Path("SAM2/annotated_images"); ANN_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1.  Load SAM once
# ------------------------------------------------------------------
sam = SAM("sam2.1_b.pt")

# ------------------------------------------------------------------
# 2.  Loop over every image
# ------------------------------------------------------------------
for img_path in tqdm.tqdm(sorted(IMG_DIR.glob("*.jpg"))):
    stem      = img_path.stem
    label_txt = LBL_DIR / f"{stem}.txt"

    if not label_txt.exists():      # no labels → skip
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    # --------------------------------------------------------------
    # 2a.  Parse the YOLO txt (class  xcen  ycen  w  h)  →  xyxy pixels
    # --------------------------------------------------------------
    boxes = []
    with open(label_txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x0 = int((xc - bw / 2) * w)
            y0 = int((yc - bh / 2) * h)
            x1 = int((xc + bw / 2) * w)
            y1 = int((yc + bh / 2) * h)
            # clip to image bounds
            boxes.append([max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)])

    if not boxes:                   # no valid boxes
        continue

    # --------------------------------------------------------------
    # 2b.  Segment with SAM (one call per image)
    # --------------------------------------------------------------
    res   = sam(img, bboxes=boxes)[0]          # Results object
    masks = res.masks.data.cpu().numpy()       # (N, H, W)  0/1

    # --------------------------------------------------------------
    # 2c.  Save each binary mask  (0-bg, 255-fg)
    # --------------------------------------------------------------
    for i, m in enumerate(masks):
        cv2.imwrite(str(MASK_DIR / f"{stem}_mask{i}.png"),
                    (m * 255).astype(np.uint8))

    # --------------------------------------------------------------
    # 2d.  Save an overlay image for quick visual check
    # --------------------------------------------------------------
    overlay = res.plot(labels=False)           # coloured masks, no text
    cv2.imwrite(str(ANN_DIR / f"{stem}_annotated.jpg"), overlay)

    # free a bit of GPU RAM on small cards
    torch.cuda.empty_cache()

print("✓ Done — masks in", MASK_DIR, "and overlays in", ANN_DIR)
