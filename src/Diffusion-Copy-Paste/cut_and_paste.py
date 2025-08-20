import logging
import os
from PIL import Image, ImageOps, ImageChops
import numpy as np
import random
import cv2

from defaults import *


def get_mask_overlap(mask1, mask2):
	'''Check if two masks overlap or not. This is determined by maximum allowed IOU between masks.
	   If IOU is less than the max allowed IOU then masks don't overlap

	Args: mask1(NumPy Array) & mask2(NumPy Array): Mask 1 & 2
	Returns: True if masks overlap else False
	'''
	m1_sum = mask1.sum()
	m2_sum = mask2.sum()
	assert m1_sum > 0 and m2_sum > 0, "One of the masks is empty, cannot compute IOU"
	intersection = np.logical_and(mask1, mask2)
	iou1 = np.sum(intersection) / m1_sum
	iou2 = np.sum(intersection) / m2_sum
	return max(iou1, iou2)


def PIL2array1C(img):
	'''Converts a PIL image to NumPy Array

	Args: img(PIL Image): Input PIL image
	Returns: NumPy Array: Converted image
	'''
	return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])


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


def prearrange_objects(objects, img_file, anno_files, label_map, w, h, scale_augment, rotation_augment, blending_list, allow_full_occlusion):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        img_file(str): Image file name
        anno_files(str): Annotation file names
        bg_file(str): Background image path 
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dont_occlude_much(bool): Generate images with occlusion
    '''
    w = WIDTH
    h = HEIGHT
    blending_list = BLENDING_LIST

    if os.path.exists(anno_files[0]):
        logging.info(f"Annotation file {anno_files[0]} already exists, skipping this image")
        return anno_files[0]

    logging.info(f"Working on annotations {anno_files} which has objects: {objects}")

    assert len(objects) > 0

    logging.info('Creating a new image now...')
	
    for attempt in range(MAX_ATTEMPTS_TO_SYNTHESIZE):
        logging.info(f'\tStarting {attempt}th attempt to synthesize this image...')
        already_syn_objs = []
        all_objects_success = True
        objs_n_masks = [] # reset the list of objects and masks to try again

        # Try to place each object that's been assigned to this image (or skip if unplaceable)
        for idx, obj in enumerate(objects):
            logging.info(f'\tStarting object {obj[0]}...')
            foreground = Image.open(obj[0]).convert('RGB')
            foreground = ImageOps.exif_transpose(foreground)  # fix orientation if needed
            mask_file =  os.path.join(os.path.dirname(obj[0]), os.path.basename(obj[0]).split('.')[0] + '_mask.png').replace('/images', '/masks')

            original_mask = Image.open(mask_file)
            mask = original_mask
            mask_arr = PIL2array1C(original_mask)
            if INVERTED_MASK:
                mask = Image.fromarray(255-mask_arr).convert('1')

            assert mask.size == foreground.size, f"Mask size {mask.size} does not match foreground size {foreground.size} for object {obj[0]}"
            
            o_w, o_h = foreground.size

            if rotation_augment:
                # logging.info(f'\t\tRotating object of size {o_w}x{o_h}...')
                rot_degrees = random.randint(-MAX_DEGREES, MAX_DEGREES)

                foreground = foreground.rotate(rot_degrees, expand=True)
                mask = mask.rotate(rot_degrees, expand=True)

                dilated_mask = Image.fromarray(cv2.dilate(mask_arr, np.ones((20,20), np.uint8), iterations=1), 'L')
                if INVERTED_MASK:
                    dilated_mask = Image.fromarray(PIL2array1C(dilated_mask)).convert('1')
                
                xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
                xmin_d, xmax_d, ymin_d, ymax_d = get_annotation_from_mask(dilated_mask)

                if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
                    raise ValueError(f"Invalid mask for object {obj[0]}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

                # logging.info(f'\t\tObject {obj[0]} mask annotation: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}')
                # logging.info(f'\t\tObject {obj[0]} dilated mask annotation: xmin={xmin_d}, ymin={ymin_d}, xmax={xmax_d}, ymax={ymax_d}')

                # logging.info(f'\t\t Foreground size before crop: {foreground.size}')
                # logging.info(f'\t\t Mask size before crop: {mask.size} | Dilated mask size before crop: {dilated_mask.size}')

                foreground = foreground.crop((xmin, ymin, xmax, ymax))
                mask = mask.crop((xmin, ymin, xmax, ymax))
                dilated_mask = dilated_mask.crop((xmin_d, ymin_d, xmax_d, ymax_d))

                # logging.info(f'\t\t Foreground size after crop: {foreground.size}')
                # logging.info(f'\t\t Mask size after crop: {mask.size} | Dilated mask size after crop: {dilated_mask.size}')

                o_w, o_h = foreground.size
                logging.info(f'\t\tObject rotated by {rot_degrees} degrees, new size: {o_w}x{o_h}')

            if o_w < 20 or o_h < 20:
                logging.warning(f'\t\tObject (after crop n rotate) is quite small!! ({o_w}x{o_h})...')

            if scale_augment:
                ACTUAL_MIN_SCALE = MIN_SCALED_DIM / min(o_w, o_h) # every object should be at least MIN_SCALED_DIM pixels in width/height
                ACTUAL_MAX_SCALE = min(w,h) / max(o_w, o_h) * 0.95 # every object should be at most min(w,h) pixels in width/height
                # the *0.9 ensures the object isn't exactly as wide/tall as the background since that may still give errors

                # scale the object so it occupies similar fraciton of the image as it did in the original foreground image
                length_scale = ((w*h) / (original_mask.size[0] * original_mask.size[1]))**0.5 
                scale = random.uniform(max(MIN_SCALE*length_scale, ACTUAL_MIN_SCALE), min(ACTUAL_MAX_SCALE, MAX_SCALE*length_scale))
                o_w, o_h = int(scale*o_w), int(scale*o_h)
                assert w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0, "Invalid object dimensions after scaling"

                foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
                mask = mask.resize((o_w, o_h), Image.LANCZOS)
                dilated_mask = dilated_mask.resize((o_w, o_h), Image.LANCZOS)
                
            # Compare current mask with all previous masks to avoid excess occlusion
            xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
            for placement_attempt in range(MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE):
                logging.info('\t\tStarting an object placement attempt...')
                x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
                y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))

                # if occlusion is allowed don't bother checking for overlap between objects
                if allow_full_occlusion:
                    break

                # Check if the object overlaps with any of the already synthesized objects
                obj_placement_is_valid = True

                trim_x_min = max(0, -x)
                trim_x_max = min(o_w, w - x)
                trim_y_min = max(0, -y)
                trim_y_max = min(o_h, h - y)

                trim_o_w = trim_x_max - trim_x_min
                trim_o_h = trim_y_max - trim_y_min

                paste_x_min = max(0, x)
                paste_x_max = paste_x_min + trim_o_w
                paste_y_min = max(0, y)
                paste_y_max = paste_y_min + trim_o_h

                assert paste_x_max <= w and paste_y_max <= h, f"Invalid paste coordinates: {paste_x_min}, {paste_y_min}, {paste_x_max}, {paste_y_max} for object {obj[0]} with size {o_w}x{o_h} at position ({x}, {y})"
                
                mask_array = np.zeros((h, w), dtype=np.uint8)
                trimmed_mask = PIL2array1C(mask)[trim_y_min:trim_y_max, trim_x_min:trim_x_max]
                mask_array[paste_y_min:paste_y_max, paste_x_min:paste_x_max] = trimmed_mask
                mask_array = np.where(mask_array > 2, True, False)

                for prev_obj in already_syn_objs:
                    overlap = get_mask_overlap(mask_array, prev_obj)
                    if overlap > MAX_OCCLUSION_IOU:
                        logging.info('\t\t\tOcclusion found, trying again...')
                        obj_placement_is_valid = False
                        break

                if obj_placement_is_valid:
                    if not allow_full_occlusion:
                        already_syn_objs.append(mask_array)
                    break
                
            else: # if we reach here, it means we could not place the object after MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE
                logging.warning(f'\t\tCould not place object {obj[0]} after {MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE} attempts')
                all_objects_success = False
                break
            
            objs_n_masks.append((obj[1], (x, y, xmin, ymin, xmax, ymax), foreground, mask, mask_array, dilated_mask))

        # If we could place all objects, then we're done trying object placements
        if all_objects_success:
            break        

    background = Image.new('RGB', (w, h), (128, 128, 128))  # create a gray background
    latent_mask = np.zeros((h, w), dtype=np.uint8)  # create an empty mask

    synth_images = []
    synth_masks = []
    for i in range(len(blending_list)):
        synth_images.append(background.copy())
        synth_masks.append(latent_mask.copy())

    logging.info('Blending objects now...')
    # Start pasting and blending objects
    for idx, obj_n_mask in enumerate(objs_n_masks):
        obj_class, (x, y, xmin, ymin, xmax, ymax), foreground, mask, mask_array, dilated_mask = obj_n_mask
        # Paste image on different background copies according to the different blending modes
        for i in range(len(blending_list)):
            if blending_list[i] == 'none' or blending_list[i] == 'motion':
                synth_images[i].paste(foreground, (x, y), mask)
                synth_masks[i] = cv2.bitwise_or(mask_array.astype(np.uint8), synth_masks[i])

            elif blending_list[i] == 'gaussian':
                blurred_mask = Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2))
                synth_images[i].paste(foreground, (x, y), blurred_mask)
                synth_masks[i] = cv2.bitwise_or(mask_array.astype(np.uint8), synth_masks[i])

            elif blending_list[i] == 'box':
                blurred_mask = Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3)))
                synth_images[i].paste(foreground, (x, y), blurred_mask)
                synth_masks[i] = cv2.bitwise_or(mask_array.astype(np.uint8), synth_masks[i])
        
        # Save annotations in text file
        class_num = [key for key in label_map if label_map[key] == obj_class][0]
        xmin = max(1, x+xmin)
        xmax = min(w, x+xmax)
        ymin = max(1, y+ymin)
        ymax = min(h, y+ymax)
        string = f"{class_num} {(xmin+xmax)/(2*w)} {(ymin+ymax)/(2*h)} {(xmax-xmin)/w} {(ymax-ymin)/h}\n"
        for anno_file in anno_files:
            if not os.path.exists(anno_file):
                os.system(f"touch {anno_file}")  # create the file if it doesn't exist
            logging.info(f'\t\tWriting annotation for object {obj_class} at {x}, {y} with mask {xmin, ymin, xmax, ymax} to {anno_file} if it is in {SELECTED_CLASSES}')
            if obj_class in SELECTED_CLASSES:
                with open(anno_file, "a") as f:
                    f.write(string)
    return synth_images, synth_masks
