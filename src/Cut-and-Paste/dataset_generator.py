import argparse
import yaml
from collections import namedtuple

import glob
import os
import shutil
from multiprocessing import Pool
from functools import partial
import signal
import time

import math
import numpy as np
import random
from PIL import Image
import cv2

from defaults import *
# from pb_master.pb import *
from pb_master.pb import create_mask, poisson_blend
from pyblur_master.pyblur import LinearMotionBlur

np.random.seed(0)
random.seed(0)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:    kerneldim (int): size of the kernel used in motion blurring
    Returns: random angle (int)
    """ 
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint=False)
    return int(np.random.choice(validLineAngles))

def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate blurring caused due to motion of camera.

    Args: img(NumPy Array): Input image with 3 channels
    Returns: Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLength = np.random.choice(lineLengths)
    lineType = np.random.choice(lineTypes)
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed 
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes 
       don't overlap

    Args: a(Rectangle) & b(Rectangle): Bounding box 1 & 2
    Returns: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    return (dx>=0) and (dy>=0) and (float(dx*dy) > MAX_OCCLUSION_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin))

def mask_overlap(mask1, mask2):
    '''Check if two masks overlap or not. This is determined by maximum allowed IOU between masks.
       If IOU is less than the max allowed IOU then masks don't overlap

    Args: mask1(NumPy Array) & mask2(NumPy Array): Mask 1 & 2
    Returns: True if masks overlap else False
    '''
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return False
    intersection = np.logical_and(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou > MAX_OCCLUSION_IOU

def read_mask_from_file(mask_file):
    '''Read mask from file and return it as a NumPy array

    Args: mask_file(string): Path of the mask file
    Returns: NumPy Array: Mask read from the file
    '''
    assert os.path.exists(mask_file), f"Mask file does not exist: {mask_file}"
    mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    if INVERTED_MASK:
        mask = 255 - mask
    return mask

def get_annotation_from_mask(mask, scale=1.0):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert len(np.where(rows)[0]) > 0, f"Found an all black mask file: {mask}"

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)

def write_yaml_file(exp_dir, labels):
    '''Writes the .yaml for YOLO training.

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = sorted(set(labels))
    yaml_filename = f'{exp_dir.split("/")[-2]}.yaml'
    yaml_path = os.path.join('/home/data', yaml_filename)

    data = {
        'path': str(exp_dir),
        'train': os.path.join(exp_dir, 'train'),
        'val': os.path.join(exp_dir, 'val'),
        'test': os.path.join(exp_dir, 'test'),
        'nc': len(unique_labels),
        'names': {i: label for i, label in enumerate(unique_labels)}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


def keep_selected_labels(img_files, labels):
    '''Filters image files and labels to only retain those that are selected. Useful when one doesn't 
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponidng to each imahe in above list
    '''
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in range(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args: img(PIL Image): Input PIL image
    Returns: NumPy Array: Converted image
    '''
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args: img(PIL Image): Input PIL image
    Returns: NumPy Array: Converted image
    '''
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def create_image_anno_wrapper(args, w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dont_occlude_much=False):
   ''' Wrapper used to pass params to workers '''
   return create_image_anno(*args, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=blending_list, dont_occlude_much=dont_occlude_much)

def create_image_anno(objects, distractor_objects, img_file, anno_file, bg_file,  w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dont_occlude_much=False):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path 
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dont_occlude_much(bool): Generate images with occlusion
    '''
    assert 'none' in img_file, "Base image file should contain 'none' blending mode in its name"
    
    if os.path.exists(anno_file):
        print(f"Annotation file {anno_file} already exists, skipping this image")
        return anno_file

    print(f"Working on {img_file}")
    
    all_objects = objects + distractor_objects
    assert len(all_objects) > 0

    for attempt in range(MAX_ATTEMPTS_TO_SYNTHESIZE):
        print('\tAnother complete attempt')

        already_syn_objs = []
        all_objects_success = True
        objs_n_masks = [] # reset the list of objects and masks to try again
        # try to place each object that's been assigned to this image (or skip if unplaceable)
        for idx, obj in enumerate(all_objects):
            print('\t\tStarting new object')
            foreground = Image.open(obj[0])
            mask_file =  os.path.join(os.path.dirname(obj[0]), os.path.basename(obj[0]).split('.')[0] + '_mask.png').replace('/images', '/masks')

            mask = Image.open(mask_file)
            if INVERTED_MASK:
                mask = Image.fromarray(255-PIL2array1C(mask)).convert('1')

            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
                raise ValueError(f"Invalid mask for object {obj[0]}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

            foreground = foreground.crop((xmin, ymin, xmax, ymax))
            orig_w, orig_h = foreground.size
            o_w, o_h = orig_w, orig_h
            
            mask = mask.crop((xmin, ymin, xmax, ymax))

            if scale_augment:
                ACTUAL_MIN_SCALE = MIN_SCALED_DIM / min(orig_w, orig_h) # every object should be at least MIN_SCALED_DIM pixels in width/height
                ACTUAL_MAX_SCALE = min(w,h) / max(orig_w, orig_h) # every object should be at most min(w,h) pixels in width/height
                scale = random.uniform(max(MIN_SCALE, ACTUAL_MIN_SCALE), min(ACTUAL_MAX_SCALE, MAX_SCALE))
                o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                assert w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0, "Invalid object dimensions after scaling"
                foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
                mask = mask.resize((o_w, o_h), Image.LANCZOS)

            if rotation_augment:
                max_degrees = MAX_DEGREES  
                while True:
                    rot_degrees = random.randint(-max_degrees, max_degrees)
                    foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                    mask_tmp = mask.rotate(rot_degrees, expand=True)
                    o_w, o_h = foreground_tmp.size
                    if  w-o_w > 0 and h-o_h > 0:
                        break
                mask = mask_tmp
                foreground = foreground_tmp

            # Compare current mask with all previous masks to avoid excess occlusion
            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            for placement_attempt in range(MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE):
                print('\t\tAnother object placement attempt')
                x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
                y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))

                # if occlusion is allowed don't bother checking for overlap between objects
                if not dont_occlude_much:
                    break

                # Check if the object overlaps with any of the already synthesized objects
                obj_placement_is_valid = True
                for prev_obj in already_syn_objs:
                    mask_array = np.pad(PIL2array1C(mask), ((h-y+o_h, y), (x, w-x+o_w)), mode='constant', constant_values=0)
                    if mask_overlap(mask_array, prev_obj):
                        print('\t\t\tOcclusion found, trying again')
                        obj_placement_is_valid = False
                        break

                if obj_placement_is_valid:
                    break
                
            else: # if we reach here, it means we could not place the object after MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE
                print(f'\t\t\tCould not place object {obj[0]} after {MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE} attempts')
                all_objects_success = False
                break
                
            if dont_occlude_much:
            #     already_syn.append(mask.data.squeeze().cpu().numpy())
                already_syn_objs.append(mask_array)
            obj_label = obj[1]
            objs_n_masks.append((obj_label, (x, y, xmin, ymin, xmax, ymax), foreground, mask))

        # If we could place all objects, then we're done trying object placements
        if all_objects_success:
            break        
    
    background = Image.open(bg_file)
    background = background.resize((w, h), Image.LANCZOS)

    synth_images = []
    for i in range(len(blending_list)):
        synth_images.append(background.copy())

    # Start pasting and blending objects
    for idx, obj_n_mask in enumerate(objs_n_masks):
        print('\tBlending object now')
        start_time = time.time()
        
        obj_label, (x, y, xmin, ymin, xmax, ymax), foreground, mask = obj_n_mask
        # Paste image on different background copies according to the different blending modes
        for i in range(len(blending_list)):
            if blending_list[i] == 'none' or blending_list[i] == 'motion':
                synth_images[i].paste(foreground, (x, y), mask)
            elif blending_list[i] == 'poisson':
                offset = (y, x)
                img_mask = PIL2array1C(mask)
                img_src = PIL2array3C(foreground).astype(np.float64)
                img_target = PIL2array3C(synth_images[i])
                img_mask, img_src, offset_adj = create_mask(img_mask.astype(np.float64),
                                                            img_target, img_src, offset=offset)
                background_array = poisson_blend(img_mask, img_src, img_target,
                                    method='normal', offset_adj=offset_adj)
                synth_images[i] = Image.fromarray(background_array, 'RGB') 
            elif blending_list[i] == 'gaussian':
                synth_images[i].paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))
            elif blending_list[i] == 'box':
                synth_images[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))
        print(f'\tBlending took {time.time() - start_time} seconds')

        # Save annotations in text file
        labels = [obj[1] for obj in objects]
        label_num = labels.index(obj_label)
        xmin = max(1, x+xmin)
        xmax = min(w, x+xmax)
        ymin = max(1, y+ymin)
        ymax = min(h, y+ymax)
        string = f"{label_num} {(xmin+xmax)/(2*w)} {(ymin+ymax)/(2*h)} {(xmax-xmin)/w} {(ymax-ymin)/h}\n"
        with open(anno_file, "a") as f:
            f.write(string)
        
    for i in range(len(blending_list)):
        # blend all objects at once for motion blur
        if blending_list[i] == 'motion':
            synth_images[i] = LinearMotionBlur3C(PIL2array3C(synth_images[i]))
        synth_images[i].save(img_file.replace('none', blending_list[i]))
   
def gen_syn_data(img_files, labels, img_dir, anno_dir, scale_augment, rotation_augment, dont_occlude_much, add_distractors):
    '''Creates list of objects and distrctor objects to be pasted on what images.
       Spawns worker processes and generates images according to given params

    Args:
        img_files(list): List of image files
        labels(list): List of labels for each image
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        dont_occlude_much(bool): Generate images with occlusion
        add_distractors(bool): Add distractor objects whose annotations are not required 
    '''
    print('Collecting background images...')
    w = WIDTH
    h = HEIGHT
    background_dir = BACKGROUND_DIR
    background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))

    print(f"Number of background images : {len(background_files)}")
    img_label_pairs = list(zip(img_files, labels))
    random.shuffle(img_label_pairs)

    if add_distractors:
        with open(DISTRACTOR_LIST_FILE) as f:
            distractor_labels = [x.strip() for x in f.readlines()]

        distractor_list = []
        for distractor_label in distractor_labels:
            distractor_list += glob.glob(os.path.join(DISTRACTOR_DIR, distractor_label, DISTRACTOR_GLOB_STRING))

        distractor_files = list(zip(distractor_list, len(distractor_list)*[None]))
        random.shuffle(distractor_files)
    else:
        distractor_files = []
    print(f"List of distractor files collected: {distractor_files}")

    idx = 0
    img_files = []
    anno_files = []
    params_list = []
    while len(img_label_pairs) > 0:
        # Get list of objects
        objects = []
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(img_label_pairs))
        for i in range(n):
            objects.append(img_label_pairs.pop())
        
        # Get list of distractor objects 
        distractor_objects = []
        if add_distractors:
            n = min(random.randint(MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS), len(distractor_files))
            for i in range(n):
                distractor_objects.append(random.choice(distractor_files))
            print(f"Chosen distractor objects: {distractor_objects}")

        idx += 1
        bg_file = random.choice(background_files)
        
        img_file = os.path.join(img_dir, f'{idx}_none.jpg')
        anno_file = os.path.join(anno_dir, f'{idx}.txt')
        params = (objects, distractor_objects, img_file, anno_file, bg_file)
        params_list.append(params)
        img_files.append(img_file)
        anno_files.append(anno_file)
        
        break # we only want to try one example

    for param in params_list:
        create_image_anno_wrapper(param, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dont_occlude_much=dont_occlude_much)

    # partial_func = partial(create_image_anno_wrapper, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dont_occlude_much=dont_occlude_much) 
    # p = Pool(NUMBER_OF_WORKERS, init_worker)
    # try:
    #     p.map(partial_func, params_list)
    # except KeyboardInterrupt:
    #     print("....\nCaught KeyboardInterrupt, terminating workers")
    #     p.terminate()
    # else:
    #     p.close()
    # p.join()
    return img_files, anno_files

def init_worker():
    ''' Catch Ctrl+C signal to terminate workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args '''
    print('Collecting image file names...')
    img_files = args.num * glob.glob(os.path.join(args.root, '*', 'images', '*'))
    random.shuffle(img_files)
    labels = [img_file.split('/')[-3] for img_file in img_files]
    write_yaml_file(args.exp, labels)

    if args.selected:
       img_files, labels = keep_selected_labels(img_files, labels)

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)

    anno_dir = os.path.join(args.exp, 'labels')
    img_dir = os.path.join(args.exp, 'images')
    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    
    syn_img_files, anno_files = gen_syn_data(img_files, labels, img_dir, anno_dir, args.scale, args.rotation, args.dont_occlude_much, args.add_distractors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("exp",
      help="The directory where images and annotation lists will be created.")
    parser.add_argument("--selected",
      help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to add scale augmentation.", action="store_false")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to add rotation data augmentation.", action="store_false")
    parser.add_argument("--num",
      help="Number of times each image will be in dataset", default=1, type=int)
    parser.add_argument("--dont_occlude_much",
      help="Add objects without too much occlusion (as defined by MAX_OCCLUSION_IOU). Default is to produce arbitrarily high occlusions (faster)", action="store_true")
    parser.add_argument("--add_distractors",
      help="Add distractors objects. Default is to not use distractors", action="store_true")
    args = parser.parse_args()
    
    generate_synthetic_dataset(args)