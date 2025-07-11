from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import shutil
import datetime

IMG_PATH = "data/synthetic/*/images/*.jpg"

def show_bbox(image_path):
    # convert image path to label path
    
    label_path = image_path.replace('/images/', '/darknet/').replace('.jpg', '.txt')

    # Open the image and create ImageDraw object for drawing
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

now = str(datetime.datetime.now())
if not os.path.exists('data/processed/' + now):
    for folder in ['images', 'labels']:
        for split in ['train', 'val', 'test']:
            os.makedirs(f'data/processed/{now}/{folder}/{split}')