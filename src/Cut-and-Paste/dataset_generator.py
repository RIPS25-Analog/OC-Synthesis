import argparse
from matplotlib import pyplot as plt
import yaml
from collections import namedtuple

import glob
import os
from multiprocessing import Pool
from functools import partial
import signal
import time

import math
import numpy as np
import random
from PIL import Image, ImageOps
import cv2

from defaults import *
from pyblur_master.pyblur import LinearMotionBlur

FIRST_TIME = True
seed = 0
np.random.seed(seed)
random.seed(seed)

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

def trim_img_n_mask(img_target, img_source, img_mask, offset):
	'''Creates a mask for the source image to be blended on the target image.
	   The mask is created by padding the source image mask with zeros according to the x, y offsets
		Also crops the source image to fit in the target image
	'''
	x, y = offset
	h_mask, w_mask = img_mask.shape
	h_target, w_target, _ = img_target.shape

	BOUNDARY_MARGIN = 2
	hd0 = max(BOUNDARY_MARGIN, -y)
	wd0 = max(BOUNDARY_MARGIN, -x)

	hd1 = h_mask - max(h_mask + y - h_target, 0) - BOUNDARY_MARGIN
	wd1 = w_mask - max(w_mask + x - w_target, 0) - BOUNDARY_MARGIN

	mask = np.zeros((h_mask, w_mask))
	mask[img_mask > 0] = 1

	mask = mask[hd0:hd1, wd0:wd1]
	src = img_source[hd0:hd1, wd0:wd1]

	# fix offset
	offset_adj = (max(x, 0), max(y, 0))

	# remove edge from the mask so that we don't have to check the
	# edge condition
	mask[:, -1] = 0
	mask[:, 0] = 0
	mask[-1, :] = 0
	mask[0, :] = 0

	return src, mask, offset_adj

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

def write_yaml_file(exp_dir, labels, label_map):
	'''Writes the .yaml for YOLO training.

	Args:
		exp_dir(string): Experiment directory where all the generated images, annotation and imageset
						 files will be stored
		labels(list): List of labels. This will be useful while training an object detector
	'''
	unique_labels = sorted(set(labels))
	yaml_filename = f'{'_'.join(exp_dir.split("/")[-2:])}.yaml' # join raw_data and processed_data name
	yaml_path = os.path.join('/home/data', yaml_filename)
	
	data = {
		'path': str(exp_dir),
		'train': os.path.join(exp_dir, 'train'),
		'val': os.path.join(exp_dir, 'val'),
		'test': os.path.join(exp_dir, 'test'),
		'nc': len(unique_labels),
		'names': label_map,
	}

	with open(yaml_path, 'w') as f:
		yaml.dump(data, f, sort_keys=False)

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
	try:
		create_image_anno(*args, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=blending_list, dont_occlude_much=dont_occlude_much)
	except Exception as e:
		logging.exception(f"Error while creating image annotation: {e}")
		if not PARALLELIZE:
			exit()

def create_image_anno(objects, distractor_objects, img_file, anno_file, bg_file, label_map, w=WIDTH, h=HEIGHT, scale_augment=False, rotation_augment=False, blending_list=['none'], dont_occlude_much=False):
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
		logging.info(f"Annotation file {anno_file} already exists, skipping this image")
		return anno_file

	logging.info(f"Working on annotation {anno_file} which has objects: {objects}")

	all_objects = objects + distractor_objects
	assert len(all_objects) > 0

	logging.info('Creating a new image now...')
	
	for attempt in range(MAX_ATTEMPTS_TO_SYNTHESIZE):
		logging.info(f'\tStarting {attempt}th attempt to synthesize this image...')
		start_time = time.time()
		already_syn_objs = []
		all_objects_success = True
		objs_n_masks = [] # reset the list of objects and masks to try again
		# try to place each object that's been assigned to this image (or skip if unplaceable)
		for idx, obj in enumerate(all_objects):
			logging.info(f'\tStarting object {obj[0]}...')
			foreground = Image.open(obj[0]).convert('RGB')
			foreground = ImageOps.exif_transpose(foreground)  # fix orientation if needed
			mask_file =  os.path.join(os.path.dirname(obj[0]), os.path.basename(obj[0]).split('.')[0] + '_mask.png').replace('/images', '/masks')

			mask = Image.open(mask_file)
			if INVERTED_MASK:
				mask = Image.fromarray(255-PIL2array1C(mask)).convert('1')

			dilated_foreground = foreground.copy()

			assert mask.size == foreground.size, f"Mask size {mask.size} does not match foreground size {foreground.size} for object {obj[0]}"
			
			o_w, o_h = foreground.size

			if rotation_augment:
				# logging.info(f'\t\tRotating object of size {o_w}x{o_h}...')
				rot_degrees = random.randint(-MAX_DEGREES, MAX_DEGREES)

				foreground = foreground.rotate(rot_degrees, expand=True)
				dilated_foreground = dilated_foreground.rotate(rot_degrees, expand=True)
				mask = mask.rotate(rot_degrees, expand=True)

				dilated_mask = Image.fromarray(cv2.dilate(PIL2array1C(mask), np.ones((20,20), np.uint8), iterations=1), 'L')
				if INVERTED_MASK:
					dilated_mask = Image.fromarray(PIL2array1C(dilated_mask)).convert('1')
				
				xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
				xmin_d, xmax_d, ymin_d, ymax_d = get_annotation_from_mask(dilated_mask)

				if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
					raise ValueError(f"Invalid mask for object {obj[0]}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

				# logging.info(f'\t\tObject {obj[0]} mask annotation: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}')
				# logging.info(f'\t\tObject {obj[0]} dilated mask annotation: xmin={xmin_d}, ymin={ymin_d}, xmax={xmax_d}, ymax={ymax_d}')

				# logging.info(f'\t\t Foreground size before crop: {foreground.size} | Dilated foreground size before crop: {dilated_foreground.size}')
				# logging.info(f'\t\t Mask size before crop: {mask.size} | Dilated mask size before crop: {dilated_mask.size}')
			
				foreground = foreground.crop((xmin, ymin, xmax, ymax))
				dilated_foreground = dilated_foreground.crop((xmin_d, ymin_d, xmax_d, ymax_d))
				mask = mask.crop((xmin, ymin, xmax, ymax))
				dilated_mask = dilated_mask.crop((xmin_d, ymin_d, xmax_d, ymax_d))

				# logging.info(f'\t\t Foreground size after crop: {foreground.size} | Dilated foreground size after crop: {dilated_foreground.size}')
				# logging.info(f'\t\t Mask size after crop: {mask.size} | Dilated mask size after crop: {dilated_mask.size}')

				o_w, o_h = foreground.size
				logging.info(f'\t\tObject rotated by {rot_degrees} degrees, new size: {o_w}x{o_h}')

			if o_w < 20 or o_h < 20:
				logging.warning(f'\t\tObject (after crop n rotate) is quite small!! ({o_w}x{o_h})...')

			if scale_augment:
				ACTUAL_MIN_SCALE = MIN_SCALED_DIM / min(o_w, o_h) # every object should be at least MIN_SCALED_DIM pixels in width/height
				ACTUAL_MAX_SCALE = min(w,h) / max(o_w, o_h) # every object should be at most min(w,h) pixels in width/height
				# the *0.9 ensures the object isn't exactly as wide/tall as the background since that may still give errors
				scale = random.uniform(max(MIN_SCALE, ACTUAL_MIN_SCALE), min(ACTUAL_MAX_SCALE, MAX_SCALE))*0.9
				o_w, o_h = int(scale*o_w), int(scale*o_h)
				assert w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0, "Invalid object dimensions after scaling"

				foreground = foreground.resize((o_w, o_h), Image.LANCZOS)
				dilated_foreground = dilated_foreground.resize((o_w, o_h), Image.LANCZOS)
				mask = mask.resize((o_w, o_h), Image.LANCZOS)
				dilated_mask = dilated_mask.resize((o_w, o_h), Image.LANCZOS)
				
			# Compare current mask with all previous masks to avoid excess occlusion
			xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
			for placement_attempt in range(MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE):
				logging.info('\t\tStarting an object placement attempt...')
				x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
				y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))

				# if occlusion is allowed don't bother checking for overlap between objects
				if not dont_occlude_much:
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
					if dont_occlude_much:
						already_syn_objs.append(mask_array)
					break
				
			else: # if we reach here, it means we could not place the object after MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE
				logging.warning(f'\t\tCould not place object {obj[0]} after {MAX_OBJECTWISE_ATTEMPTS_TO_SYNTHESIZE} attempts')
				all_objects_success = False
				break
			
			objs_n_masks.append((obj[1], (x, y, xmin, ymin, xmax, ymax), foreground, dilated_foreground, mask, dilated_mask))

		# If we could place all objects, then we're done trying object placements
		if all_objects_success:
			break        
	
	background = Image.open(bg_file).convert('RGB')
	background = ImageOps.exif_transpose(background)  # fix orientation if needed
	background = background.resize((w, h), Image.LANCZOS)

	synth_images = []
	for i in range(len(blending_list)):
		synth_images.append(background.copy())

	logging.info('Blending objects now...')
	# Start pasting and blending objects
	for idx, obj_n_mask in enumerate(objs_n_masks):
		obj_class, (x, y, xmin, ymin, xmax, ymax), foreground, dilated_foreground, mask, dilated_mask = obj_n_mask
		# Paste image on different background copies according to the different blending modes
		for i in range(len(blending_list)):
			if blending_list[i] == 'none' or blending_list[i] == 'motion':
				synth_images[i].paste(foreground, (x, y), mask)

			elif blending_list[i] == 'poisson':
				offset = (x, y)
				target = PIL2array3C(synth_images[i])

				# source = PIL2array3C(foreground)
				# source, mask_arr, offset = trim_img_n_mask(target, source, PIL2array1C(mask), offset)
				# mask_arr = (mask_arr*255).astype(np.uint8)
				# center = (offset[0] + source.shape[1]//2, offset[1] + source.shape[0]//2)
				# mixed = cv2.seamlessClone(source.copy(), target.copy(), mask_arr, center, cv2.NORMAL_CLONE)

				source = PIL2array3C(dilated_foreground)
				source, dilated_mask_arr, offset = trim_img_n_mask(target, source, PIL2array1C(dilated_mask), offset)
				dilated_mask_arr = (dilated_mask_arr*255).astype(np.uint8)
				center = (offset[0] + source.shape[1]//2, offset[1] + source.shape[0]//2)
				mixed = cv2.seamlessClone(source.copy(), target.copy(), dilated_mask_arr, center, cv2.NORMAL_CLONE)

				global FIRST_TIME
				if not PARALLELIZE and FIRST_TIME:
					FIRST_TIME = False
					logging.info(f"SAVING MASKS FOR {anno_file} with {objects[idx][0].split('/')[-1].split('.')[0]} path")
					mask.save(f"tmp/original_mask_{objects[idx][0].split('/')[-1].split('.')[0]}.png")
					dilated_mask.save(f"tmp/dilated_mask_{objects[idx][0].split('/')[-1].split('.')[0]}.png")
					foreground.save(f"tmp/foreground_{objects[idx][0].split('/')[-1].split('.')[0]}.png")
					dilated_foreground.save(f"tmp/dilated_foreground_{objects[idx][0].split('/')[-1].split('.')[0]}.png")
					# assert 0==1

				synth_images[i] = Image.fromarray(mixed.astype(np.uint8), 'RGB')

			elif blending_list[i] == 'gaussian':
				synth_images[i].paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))

			elif blending_list[i] == 'box':
				synth_images[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))

		# Save annotations in text file
		class_num = [key for key in label_map if label_map[key] == obj_class][0]
		xmin = max(1, x+xmin)
		xmax = min(w, x+xmax)
		ymin = max(1, y+ymin)
		ymax = min(h, y+ymax)
		string = f"{class_num} {(xmin+xmax)/(2*w)} {(ymin+ymax)/(2*h)} {(xmax-xmin)/w} {(ymax-ymin)/h}\n"
		with open(anno_file, "a") as f:
			f.write(string)

	logging.info(f'Creating image with all objects took {time.time() - start_time} seconds')
	logging.info('Saving images now...')
	start_time = time.time()

	# Save images
	for i in range(len(blending_list)):
		# blend all objects at once for motion blur
		if blending_list[i] == 'motion':
			synth_images[i] = LinearMotionBlur3C(PIL2array3C(synth_images[i]))
		synth_images[i].save(img_file.replace('none', blending_list[i]))

	logging.info(f'Saving images took {time.time() - start_time} seconds')
   
def gen_syn_data(img_files, classes, img_dir, anno_dir, label_map, scale_augment, rotation_augment, dont_occlude_much, add_distractors):
	'''Creates list of objects and distrctor objects to be pasted on what images.
	   Spawns worker processes and generates images according to given params

	Args:
		img_files(list): List of image files
		classes(list): List of the object class of each image
		img_dir(str): Directory where synthesized images will be stored
		anno_dir(str): Directory where corresponding annotations will be stored
		scale_augment(bool): Add scale data augmentation
		rotation_augment(bool): Add rotation data augmentation
		dont_occlude_much(bool): Generate images with occlusion
		add_distractors(bool): Add distractor objects whose annotations are not required 
	'''
	w = WIDTH
	h = HEIGHT
	background_dir = BACKGROUND_DIR
	background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))

	logging.info(f"Number of background images : {len(background_files)}")
	image_class_pairs = list(zip(img_files, classes))
	random.shuffle(image_class_pairs)

	idx = 0
	img_files = []
	anno_files = []
	params_list = []
	while len(image_class_pairs) > 0:
		# Get list of objects
		selected_objects = []
		n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(image_class_pairs))
		for i in range(n):
			selected_objects.append(image_class_pairs.pop())
		
		logging.info(f"Chosen objects: {selected_objects}")
		
		idx += 1
		bg_file = random.choice(background_files)

		logging.info(f"Chosen background file: {bg_file}")
		
		img_file = os.path.join(img_dir, f'{idx}_none.jpg')
		anno_file = os.path.join(anno_dir, f'{idx}.txt')
		distractor_objects = [] # removed concept of distractor objects
		params = (selected_objects, distractor_objects, img_file, anno_file, bg_file, label_map)
		params_list.append(params)
		img_files.append(img_file)
		anno_files.append(anno_file)
		
		# break # if we only want to try one example
	
	if PARALLELIZE:
		logging.info(f"Parallelizing with {NUMBER_OF_WORKERS} workers")
		partial_func = partial(create_image_anno_wrapper, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dont_occlude_much=dont_occlude_much) 
		p = Pool(NUMBER_OF_WORKERS, init_worker)
		try:
			p.map(partial_func, params_list)
		except KeyboardInterrupt:
			logging.warning("....\nCaught KeyboardInterrupt, terminating workers")
			p.terminate()
		else:
			p.close()
		p.join()
	else:
		logging.info("Not parallelizing, running in serial mode")
		for param in params_list:
			create_image_anno_wrapper(param, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dont_occlude_much=dont_occlude_much)
			
	print('Generation complete!')
	logging.info("Generation complete!")
	
	return img_files, anno_files

def init_worker():
	''' Catch Ctrl+C signal to terminate workers '''
	signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
	''' Generate synthetic dataset according to given args '''
	img_files = args.num * glob.glob(os.path.join(args.root, '*', 'images', '*'))[:MAX_OBJ_IMAGES]
	random.shuffle(img_files)
	class_names = [img_file.split('/')[-3] for img_file in img_files]

	unselected_classes = set(class_names) - set(SELECTED_CLASSES)
	all_classes = SELECTED_CLASSES + list(unselected_classes)
	label_map = {i: label for i, label in enumerate(all_classes)}

	write_yaml_file(args.exp, class_names, label_map)

	if not os.path.exists(args.exp):
		os.makedirs(args.exp)

	anno_dir = os.path.join(args.exp, 'labels')
	img_dir = os.path.join(args.exp, 'images')

	if not os.path.exists(os.path.join(anno_dir)):
		os.makedirs(anno_dir)
	if not os.path.exists(os.path.join(img_dir)):
		os.makedirs(img_dir)
	
	syn_img_files, anno_files = gen_syn_data(img_files, class_names, img_dir, anno_dir, label_map, args.scale, args.rotation, args.dont_occlude_much, args.add_distractors)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Create dataset with different augmentations",
									  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("root",
	  help="The root directory which contains the images and annotations.")
	parser.add_argument("exp",
	  help="The directory where images and annotation lists will be created.")
	parser.add_argument("--selected",
	  help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
	parser.add_argument("--scale",
	  help="Add scale augmentation.Default is to add scale augmentation.", action="store_true")
	parser.add_argument("--rotation",
	  help="Add rotation augmentation.Default is to add rotation data augmentation.", action="store_true")
	parser.add_argument("--num",
	  help="Number of times each image will be in dataset", default=1, type=int)
	parser.add_argument("--dont_occlude_much",
	  help="Add objects without too much occlusion (as defined by MAX_OCCLUSION_IOU). Default is to produce arbitrarily high occlusions (faster)", action="store_true")
	parser.add_argument("--add_distractors",
	  help="Add distractors objects. Default is to not use distractors", action="store_true")
	parser.add_argument("--dont_parallelize",
	  help="Run the dataset generation in parallel. Default is to run in serial mode", action="store_true")
	parser.add_argument("--max_obj_images",
	  help="Maximum number of object images to use for each synthetic image", default=int(1e6), type=int)
	args = parser.parse_args()

	PARALLELIZE = not args.dont_parallelize
	MAX_OBJ_IMAGES = args.max_obj_images

	if not os.path.exists(args.exp):
		os.makedirs(args.exp)
		
	import logging
	logger = logging.getLogger(__name__)
	logging.basicConfig(filename=f"{args.exp}/dataset_generator.log", encoding='utf-8', level=logging.DEBUG)
	logger.setLevel(logging.INFO)
	logging.getLogger('PIL').setLevel(logging.WARNING)
	logging.info(f'Generating dataset with scale={args.scale}, rotation={args.rotation}, num={args.num}, dont_occlude_much={args.dont_occlude_much}, add_distractors={args.add_distractors}')

	generate_synthetic_dataset(args)