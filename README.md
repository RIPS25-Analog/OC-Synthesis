# RIPS25-AnalogDevices-ObjectDetection

## Setup instructions:

```
python -m venv env
source env/bin/activate
pip install ultralytics, wandb
wandb login  # put API key
```

## Data file structure

Create a data directory like ```/home/data```, with subfolders called ```raw``` and ```processed```

### Real annotated data

The real annotated data should be contain images and labels within train-val-test splits:

```
└── /home/data/processed/[dataset_name]
    ├── train/
    |   ├── images/         # images of the same file type (e.g: 4_5.png, 15_10.png)
    |   └── labels/         # labels in darknet format (class_id, x, y, w, h) with same filenames as corresponding images (e.g: 4_5.txt, 15_10.txt)
    ├── val/                # same subfoldering as above
    └── test/               # same subfoldering as above
```

### 2D Approaches: Foreground Object & Background Data

Cut-n-Paste and Diffusion require a folder with images of foreground objects and a mask for each image, as well as a folder with background images:

``` 
└── /home/data/raw/[dataset_name]
    ├── foreground_objects/
    |   ├── class_name_1
    |   |   ├── images              # images of the same file type (e.g: 4_5.png, 15_10.png)
    |   |   └── masks               # masks with same filenames as correspoonding images plus _mask attached at end (e.g: 4_5_mask.png, 15_10_mask.png)
    |   ├── ...
    |   └── class_name_8
    |       ├── images
    |       └── masks
    └── backgrounds/                # backgrounds of the same file type (e.g: 1.png, 2.png)
```

### 3D Approaches

## Running scripts

To finetune YOLO on a specific data file: 

``` python src/finetune_YOLO.py --data data/openimages-basic-v0.yaml --epochs 300 --freeze 23 ```

To evaluate a particular YOLO finetuning run: 

``` python src/evaluate_YOLO.py --run runs/openimages-basic-v0/train2 ```

To run hyperopt for YOLO finetune on a specific data file: 

``` python src/hyperparam_opt_YOLO.py --data data/openimages-basic-v0.yaml --epoch 100 ```

To generate Cut n Paste dataset: 

``` python src/Cut-and-Paste/dataset_generator.py --n_images 100 /home/data/pace/toycar_can_v2-extra/foreground_objects/ /home/data/processed/cnp-pace/test-100 ```

## Folder structure

```
├── repo-name/src/                      # All code
|   ├── finetune_YOLO.py
|   ├── evaluate_YOLO.py
|   └── fetch-openimages-data.ipynb     # downloads class-specific images from Google's Open Images dataset into data/raw/, processes them into data/processed/
├── /home/wandb-runs/                   # Saved logs and results, organized into subfolders by run ID
└── /home/data/
    ├── raw/                            # Original custom object images or backgrounds
    |   ├── kaggle_v0/
    |   ├── simple-backgrounds-v0/
    |   └── 3d-objects-v0/
    ├── processed/                      # Final dataset, organized into subfolders by different synthetic methods/versions
    |   ├── kaggle-cnp-v0/
    |   |   └── synthetic-1065/         # described data mix and number of total images in each dataset
    |   |   |   ├── train/
    |   |   |   |   ├── images/
    |   |   |   |   └── labels/
    |   |   |   ├── val/                # same subfoldering as above: images and labels
    |   |   |   └── test/               # same subfoldering as above: images and labels
    |   └── cut-and-paste-v0/           # All datasets follow same structure
    ├── cnp-v0-1065.yaml/              # YAML listing classes and paths to train-val-test(all cnp-v0) directories with 1065 images
    └── cnp-v0-4985-mech-val.yaml/     # YAML listing classes and paths to train(cnp-v0)/val(mech)/test(mech) directories
        
```
