# Code & README adapted from the paper [Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://arxiv.org/abs/1708.01642) and its original [repository](https://github.com/debidatta/syndata-generation) by [Dwibedi Datta](https://github.com/debidatta)

## Pyblur code was taken from the [pyblur repository](https://github.com/lospooky/pyblur) by [Simone Cirillo](https://github.com/lospooky)

This code is used to generate synthetic scenes for the task of instance/object detection. Given images of objects in isolation from multiple views and some background scenes, it generates full scenes with multiple objects and annotations files which can be used to train an object detector.

## Running the Script
```
usage: dataset_generator.py [-h] [--n_images N_IMAGES] [--dont_scale] [--dont_rotate] [--allow_full_occlusion]
                            [--dont_add_distractors] [--dont_parallelize] root exp

Create dataset with different augmentations

positional arguments:
  root                  The root directory which contains the images and annotations.
  exp                   The directory where images and annotation lists will be created.

options:
  -h, --help            show this help message and exit
  --n_images N_IMAGES   Number of images to generate (divided by 5 for different blending modes) (default: 10000)
  --dont_scale          Remove scale augmentation. Default is to add scale augmentation. (default: False)
  --dont_rotate         Remove rotation augmentation. Default is to add rotation data augmentation. (default: False)
  --allow_full_occlusion
                        Allow complete occlusion between objects (faster). Default is to avoid high occlusions (as defined by MAX_OCCLUSION_IOU) (default: False)
  --dont_add_distractors
                        Don't add distractors objects. Default is to add distractors (default: False)
  --dont_parallelize    Run the dataset generation in serial mode. Default is to run in parallel mode (default: False)
```