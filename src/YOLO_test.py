import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import glob

def show_bbox(image_path):
    # convert image path to label path
    label_path = image_path.replace('/images/', '/darknet/')
    label_path = label_path.replace('.jpg', '.txt')

    # Open the image and create ImageDraw object for drawing
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    with open(label_path, 'r') as f:
        for line in f.readlines():
            # Split the line into five values
            label, x, y, w, h = line.split(' ')

            # Convert string into float
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            # Convert center position, width, height into
            # top-left and bottom-right coordinates
            W, H = image.size
            x1 = (x - w/2) * W
            y1 = (y - h/2) * H
            x2 = (x + w/2) * W
            y2 = (y + h/2) * H

            # Draw the bounding box with red lines
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 0, 0), # Red in RGB
                           width=5)             # Line width
    image.show()


def get_filenames(folder):
    filenames = set()
    
    for path in glob.glob(os.path.join(folder, '*.jpg')):
        # Extract the filename
        filename = os.path.split(path)[-1]        
        filenames.add(filename)

    return filenames


# Dog and cat image filename sets
dog_images = get_filenames('download/dog/images')
cat_images = get_filenames('download/cat/images')

# Check for duplicates
duplicates = dog_images & cat_images

# Show the images from the duplicated filenames
for file in duplicates:
    for animal in ['cat', 'dog']:
        img = Image.open(f'download/{animal}/images/{file}')
        print(img)
        plt.imshow(img)
        plt.title(f"{animal}: {file}")
        plt.axis('off')
        plt.show(block=True)