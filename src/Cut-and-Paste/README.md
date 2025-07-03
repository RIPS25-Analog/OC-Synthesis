# SynDataGeneration 

This code is used to generate synthetic scenes for the task of instance/object detection. Given images of objects in isolation from multiple views and some background scenes, it generates full scenes with multiple objects and annotations files which can be used to train an object detector. The approach used for generation works welll with region based object detection methods like [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn).

## Pre-requisites 
1. OpenCV (pip install opencv-python)
2. PIL (pip install Pillow)
3. Poisson Blending (Follow instructions [here](https://github.com/yskmt/pb)
4. PyBlur (pip install pyblur)

To be able to generate scenes this code assumes you have the object masks for all images. There is no pre-requisite on what algorithm is used to generate these masks as for different applications different algorithms might end up doing a good job. However, we recommend [Pixel Objectness with Bilinear Pooling](https://github.com/debidatta/pixelobjectness-bp) to automatically generate these masks. If you want to annotate the image manually we recommend GrabCut algorithms([here](https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py), [here](https://github.com/cmuartfab/grabcut), [here](https://github.com/daviddoria/GrabCut))

## Setting up Defaults
The first section in the defaults.py file contains paths to various files and libraries. Set them up accordingly.

The other defaults refer to different image generating parameters that might be varied to produce scenes with different levels of clutter, occlusion, data augmentation etc. 

## Running the Script
```
python dataset_generator.py [-h] [--selected] [--scale] [--rotation]
                            [--num NUM] [--dontocclude] [--add_distractors]
                            root exp

Create dataset with different augmentations

positional arguments:
  root               The root directory which contains the images and
                     annotations.
  exp                The directory where images and annotation lists will be
                     created.

optional arguments:
  -h, --help         show this help message and exit
  --selected         Keep only selected instances in the test dataset. Default
                     is to keep all instances in the roo directory.
  --scale            Add scale augmentation.Default is to not add scale
                     augmentation.
  --rotation         Add rotation augmentation.Default is to not add rotation
                     augmentation.
  --num NUM          Number of times each image will be in dataset
  --dontocclude      Add objects without occlusion. Default is to produce
                     occlusions
  --add_distractors  Add distractors objects. Default is to not use
                     distractors
```

## Training an object detector
The code produces all the files required to train an object detector. The format is directly useful for Faster R-CNN but might be adapted for different object detectors too. The different files produced are:
1. __labels.txt__ - Contains the labels of the objects being trained
2. __annotations/*.xml__ - Contains annotation files in XML format which contain bounding box annotations for various scenes
3. __images/*.jpg__ - Contain image files of the synthetic scenes in JPEG format 
4. __train.txt__ - Contains list of synthetic image files and corresponding annotation files

There are tutorials describing how one can adapt Faster R-CNN code to run on a custom dataset like:
1. https://github.com/rbgirshick/py-faster-rcnn/issues/243
2. http://sgsai.blogspot.com/2016/02/training-faster-r-cnn-on-custom-dataset.html

## Paper

The code was used to generate synthetic scenes for the paper [Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://arxiv.org/abs/1708.01642). 

If you find our code useful in your research, please consider citing:
```
@InProceedings{Dwibedi_2017_ICCV,
author = {Dwibedi, Debidatta and Misra, Ishan and Hebert, Martial},
title = {Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```

# Modern Dataset Generator

A modernized Python 3 version of the cut-and-paste synthetic dataset generator.

## Features

- Python 3 compatible with modern dependencies
- Simplified file structure requirements
- Automatic YAML config generation for training
- YOLO format annotations
- Multiprocessing support
- Configurable augmentations

## Requirements

```bash
pip install pillow opencv-python numpy pyyaml
```

## Directory Structure

### Input Structure
```
objects_dir/
├── class1/
│   ├── image1.jpg
│   ├── image1_mask.png
│   ├── image2.jpg
│   └── image2_mask.png
└── class2/
    ├── image1.jpg
    ├── image1_mask.png
    └── ...

backgrounds_dir/
├── bg1.jpg
├── bg2.jpg
└── ...
```

### Output Structure
```
output_dir/
├── images/
│   ├── synthetic_000001.jpg
│   └── ...
├── labels/
│   ├── synthetic_000001.txt
│   └── ...
└── dataset_config.yaml
```

## Usage

Basic usage:
```bash
python modern_dataset_generator.py objects_dir backgrounds_dir output_dir --num_images 1000
```

With configuration file:
```bash
python modern_dataset_generator.py objects_dir backgrounds_dir output_dir --config example_config.json --num_images 5000
```

## Configuration Options

- `width`, `height`: Output image dimensions
- `min_objects`, `max_objects`: Range of objects per image
- `min_scale`, `max_scale`: Scale augmentation range
- `max_rotation`: Maximum rotation angle in degrees
- `max_iou`: Maximum allowed IoU between objects
- `blur_probability`: Probability of applying motion blur
- `num_workers`: Number of parallel processes

## Mask Requirements

- Masks should be grayscale images with the same name as the object image plus "_mask" suffix
- White pixels (255) represent the object, black pixels (0) represent background
- Masks should have the same dimensions as the corresponding object image
