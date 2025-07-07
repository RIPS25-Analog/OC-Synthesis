# RIPS25-AnalogDevices-ObjectDetection

To create virtual environment:
```python -m venv env```

To activate:
```source env/bin/activate```

YOLO classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

YOLO train method docs: https://docs.ultralytics.com/usage/cfg/#train-settings

To finetune YOLO on a specific data file: ``` python src/finetune_YOLO.py --data_path data/openimages-basic-v0.yaml --epochs 300 --freeze 23 ```

To evaluate a particular YOLO finetuning run: ``` python src/evaluate_YOLO.py --run_path runs/openimages-basic-v0/train2 ```

## Folder structure

```
├── src/                            # All code
|   ├── finetune_YOLO.py
|   ├── evaluate_YOLO.py
|   ├── fetch-openimages-data.ipynb # Fetches images from Google's Open Images dataset
|   └── SAM_controller.ipynb        # Segments foreground images and calls dataset_generator.py to create synthetic data
├── runs/                           # Saved logs and results, organized into subfolders by run ID
└── data/
    ├── raw/                        # Original custom object samples
    ├── synthetic/                  # Generated synthetic data, organized into subfolders by different methods and versions
    |   ├── cut-and-paste-v0/
    |   ├── cut-and-paste-v1/
    |   ├── ...
    |   └── 3d-rendered-v5/
    └── processed/                  # Final train-val-test datasets, with same subfoldering structure
```
