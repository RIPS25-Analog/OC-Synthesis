# Setup instructions:

```
python -m venv env
source env/bin/activate
pip install ultralytics, wandb
wandb login  # put API key
```

# Data file structure

Create a data directory like ```/home/data```, with subfolders called ```raw``` and ```processed```

## Real annotated data

The real annotated data should be contain images and labels within train-val-test splits:

```
└── /home/data/processed/[dataset_name]
    ├── train/
    |   ├── images/         # images of the same file type (e.g: 4_5.png, 15_10.png)
    |   └── labels/         # labels in darknet format (class_id, x, y, w, h) with same filenames as corresponding images (e.g: 4_5.txt, 15_10.txt)
    ├── val/                # same subfoldering as above
    └── test/               # same subfoldering as above
```

## 2D Approaches: Foreground Object & Background Data

Cut-n-Paste and Diffusion require a folder with images of foreground objects and a mask for each image, as well as a folder with background images:

``` 
└── /home/data/raw/[dataset_name]
    ├── foreground_objects/
    |   ├── class_name_1
    |   |   ├── images              # images of the same file type (e.g: 4_5.png, 15_10.png)
    |   |   └── masks               # masks with same filenames as corresponding images plus _mask attached at end (e.g: 4_5_mask.png, 15_10_mask.png)
    |   ├── ...
    |   └── class_name_8
    └── backgrounds/                # backgrounds of the same file type (e.g: 1.png, 2.png)
```

## 3D Approaches

3D Random Placement and 3D Copy-Paste require a folder with 3d models of objects, as well as folders with HDRI backgrounds and RGBD backgrounds

``` 
└── /home/data/raw/[dataset_name]
    ├── 3d_models/
    |   ├── class_name_1            # e.g: hammers
    |   |   ├── instance1           # e.g: big red hammer
    |   |   |   ├── instance1.obj   # 3D OBJ file
    |   |   |   ├── instance1.mtl   # MTL file to map texture image to object's faces
    |   |   |   └── instance1.png   # texture for 3d object
    |   |   ├── ...
    |   |   └── instance5           
    |   ├── ...
    |   ├── ...
    |   └── class_name_8
    └── backgrounds/
        ├── HDRI                    # contains HDRI background files of .exr filetype
        └── RGBD
            ├── image               # contains RGB images of the same filetype  (e.g: 0001.jpg)
            └── depth               # contains depth images with same filenames as corresponding images (e.g: 0001.mat)
```

# Running scripts

To finetune YOLO on a specific data file: 

``` python src/finetune_YOLO.py --data data/openimages-basic-v0.yaml --epochs 300 --freeze 23 ```

To evaluate a particular YOLO finetuning run: 

``` python src/evaluate_YOLO.py --run runs/openimages-basic-v0/train2 ```

To run hyperopt for YOLO finetune on a specific data file: 

``` python src/hyperparam_opt_YOLO.py --data data/openimages-basic-v0.yaml --epoch 100 ```

To generate Cut n Paste dataset: 

``` python src/Cut-and-Paste/dataset_generator.py --n_images 100 /home/data/pace/toycar_can_v2-extra/foreground_objects/ /home/data/processed/cnp-pace/test-100 ```