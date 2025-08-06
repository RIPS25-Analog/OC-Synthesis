# RIPS25-AnalogDevices-ObjectDetection

## Setup instructions:

```
python -m venv env
source env/bin/activate
pip install ultralytics, wandb
wandb login  # put API key
```

## Running scripts

To finetune YOLO on a specific data file: 

``` python src/finetune_YOLO.py --data data/openimages-basic-v0.yaml --epochs 300 --freeze 23 ```

To evaluate a particular YOLO finetuning run: 

``` python src/evaluate_YOLO.py --run runs/openimages-basic-v0/train2 ```

To run hyperopt for YOLO finetune on a specific data file: 

``` python src/hyperparam_opt_YOLO.py --data data/openimages-basic-v0.yaml --epoch 100 ```

To generate Cut n Paste dataset: 

``` python src/Cut-and-Paste/dataset_generator.py --num 20 --dont_occlude_much /home/data/raw/kaggle_v0/ /home/data/processed/kaggle-cnp-v0/[desired-dataset-name] ```

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

## Extra info

YOLO classes: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

Google Open Images classes: https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv

YOLO train method docs: https://docs.ultralytics.com/usage/cfg/#train-settings
