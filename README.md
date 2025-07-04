# RIPS25-AnalogDevices-ObjectDetection

Setup instructions:

```
python -m venv env
source env/bin/activate
wandb login  # put API key
yolo settings wandb=true
```

YOLO classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

YOLO train method docs: https://docs.ultralytics.com/usage/cfg/#train-settings

To finetune YOLO on a specific data file: ``` python src/finetune_YOLO.py --data_path data/openimages-basic-v0.yaml --epochs 300 --freeze 23 ```

To evaluate a particular YOLO finetuning run: ``` python src/evaluate_YOLO.py --run_path runs/openimages-basic-v0/train2 ```

To test statistics for a particular dataset: ``` python src/data-stats-checker.py --data_path data/openimages-basic-v0.yaml ```

## Folder structure

```
├── src/                            # All code
|   ├── finetune_YOLO.py
|   ├── evaluate_YOLO.py
|   └── fetch-openimages-data.ipynb # downloads class-specific images from Google's Open Images dataset into data/raw/, processes them into data/processed/
├── runs/                           # Saved logs and results, organized into subfolders by run ID
└── data/
    ├── raw/                        # Original custom object samples
    |   ├── openimages-v0/
    |   ├── simple-backgrounds-v0/
    |   └── 3d-objects-v0/
    ├── intermediary/               # OPTIONAL step in generating final dataset, organized into subfolders by different synthetic methods/versions
    |   ├── openimages-basic-v0/
    |   └── cut-and-paste-v0/
    └── processed/                  # Final train-val-test datasets
        ├── openimages-basic-v0/
        |   ├── train/
        |   |   ├── images/
        |   |   └── labels/
        |   ├── val/                # same subfoldering as above: images and labels
        |   └── test/               # same subfoldering as above: images and labels
        └── cut-and-paste-v0/       # All datasets follow same structure
```