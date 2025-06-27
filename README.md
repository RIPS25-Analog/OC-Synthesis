# RIPS25-AnalogDevices-ObjectDetection

To create virtual environment:
```python -m venv env```

To activate:
```source env/bin/activate```

YOLO classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

## Folder structure

```
├── src/               # All code
├── experiments/      # Saved logs and results, organized into subfolders by run ID
├── data/
│   ├── raw/         # Original custom object samples
│   ├── synthetic/   # Generated synthetic data, organized into subfolders by different methods and versions
│   |   ├── cut-and-paste-v0/
│   |   ├── cut-and-paste-v1/
│   |   ├── ...
│   |   └── 3d-rendered-v5/
│   └── processed/   # Final train-val-test datasets
└── models/          # Saved model checkpoints
```